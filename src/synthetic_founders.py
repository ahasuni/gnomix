import json
import hashlib
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans

from src.utils import read_vcf, read_genetic_map


@dataclass
class Window:
    window_id: int
    start_idx: int
    end_idx: int  # exclusive
    start_bp: int
    end_bp: int
    start_cm: float
    end_cm: float


class SyntheticFounderGenerator:
    """
    Build synthetic ancestry-specific haplotypes from a phased reference VCF.

    Design
    ------
    1. Read phased reference VCF and subset to shared SNP scaffold
    2. Interpolate cM positions using shared genetic map
    3. Build deterministic fixed windows (for fitting local models)
    4. Fit KMeans-based local mixture models within each window
    5. Generate synthetic full-length haplotypes using the SAME breakpoint
       logic as laidataset.py:
          - number of crossovers from Poisson
          - breakpoints sampled from adjacent-SNP map probabilities
       These internal segments are variable-length
    6. Output full-length haplotypes on the shared scaffold
    """

    def __init__(
        self,
        chm: str,
        reference_vcf: str,
        genetic_map_file: str,
        ancestry_label: str,
        snp_manifest_file: str,
        config: dict,
        seed: Optional[int] = None,
    ):
        self.chm = str(chm)
        self.reference_vcf = reference_vcf
        self.genetic_map_file = genetic_map_file
        self.ancestry_label = ancestry_label
        self.snp_manifest_file = snp_manifest_file
        self.config = config
        self.seed = config.get("seed", 94305) if seed is None else seed

        # Match laidataset.py behavior by using the global NumPy RNG
        np.random.seed(self.seed)

        self.window_size_cM = float(config["model"]["window_size_cM"])

        # Optional synthetic-specific settings
        synth_cfg = config.get("synthetic", {})
        self.k_min = int(synth_cfg.get("k_min", 2))
        self.k_max = int(synth_cfg.get("k_max", 8))
        self.alpha = float(synth_cfg.get("alpha", 0.5))
        self.min_window_snps = int(synth_cfg.get("min_window_snps", 20))
        self.max_window_snps = int(synth_cfg.get("max_window_snps", 500))
        self.n_init = int(synth_cfg.get("n_init", 10))

        # Internal recombination complexity for synthetic founder generation
        self.synthetic_gen = int(synth_cfg.get("synthetic_gen", 1))

        # Loaded / derived data
        self.positions: Optional[np.ndarray] = None
        self.refs: Optional[np.ndarray] = None
        self.alts: Optional[np.ndarray] = None
        self.samples: Optional[np.ndarray] = None
        self.gt: Optional[np.ndarray] = None              # (n_snps, n_samples, 2)
        self.cm_positions: Optional[np.ndarray] = None    # (n_snps,)
        self.haplotypes: Optional[np.ndarray] = None      # (n_haps, n_snps)

        # Same objects conceptually as get_chm_info() in laidataset.py
        self.chm_length_morgans: Optional[float] = None
        self.breakpoint_probability: Optional[np.ndarray] = None  # len = n_snps - 1

        self.windows: List[Window] = []
        self.window_models: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Loading / preprocessing
    # ------------------------------------------------------------------

    def load_reference(self) -> None:
        """
        Read phased VCF and subset to the shared SNP scaffold in manifest order.

        Expected manifest columns:
            chrom, pos, ref, alt
        """
        vcf_data = read_vcf(self.reference_vcf, self.chm)

        pos = vcf_data["variants/POS"].copy()
        ref = vcf_data["variants/REF"].copy().astype(str)
        alt = vcf_data["variants/ALT"][:, 0].copy().astype(str)
        gt = vcf_data["calldata/GT"]
        samples = np.array(vcf_data["samples"])

        if gt is None:
            raise ValueError("VCF did not contain calldata/GT.")
        if gt.ndim != 3 or gt.shape[2] != 2:
            raise ValueError(f"Expected GT shape (n_snps, n_samples, 2), got {gt.shape}")

        manifest = pd.read_csv(self.snp_manifest_file, sep="\t", dtype={"chrom": str})
        required_cols = {"chrom", "pos", "ref", "alt"}
        missing = required_cols - set(manifest.columns)
        if missing:
            raise ValueError(f"SNP manifest missing required columns: {missing}")

        # Restrict manifest to this chromosome
        manifest = manifest[manifest["chrom"].astype(str).isin([self.chm, f"chr{self.chm}"])].copy()
        if len(manifest) == 0:
            raise ValueError(f"No SNPs for chromosome {self.chm} in manifest.")

        # Normalize chromosome labels
        manifest["chrom_norm"] = manifest["chrom"].astype(str).str.replace("^chr", "", regex=True)

        ref_df = pd.DataFrame({
            "chrom": np.array([str(self.chm)] * len(pos)),
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "idx": np.arange(len(pos))
        })
        ref_df["chrom_norm"] = ref_df["chrom"].astype(str).str.replace("^chr", "", regex=True)

        # Merge in manifest order
        merged = manifest.merge(
            ref_df,
            on=["chrom_norm", "pos", "ref", "alt"],
            how="inner",
            sort=False,
        )

        if len(merged) == 0:
            raise ValueError("No overlapping SNPs between manifest and VCF.")

        keep_idx = merged["idx"].to_numpy()

        self.positions = pos[keep_idx]
        self.refs = ref[keep_idx]
        self.alts = alt[keep_idx]
        self.gt = gt[keep_idx, :, :].astype(np.uint8)
        self.samples = samples

    def get_chm_info(self) -> None:
        """
        Match laidataset.py get_chm_info() logic.

        Produces:
          - chromosome length in Morgans
          - breakpoint probability across adjacent SNP intervals
          - interpolated cM position at each retained SNP
        """
        if self.positions is None:
            raise ValueError("Run load_reference() first.")

        genetic_chm = read_genetic_map(self.genetic_map_file, self.chm)

        # Same logic as laidataset.py
        self.chm_length_morgans = max(genetic_chm["pos_cm"]) / 100.0

        genomic_intervals = interp1d(
            x=genetic_chm["pos"].to_numpy(),
            y=genetic_chm["pos_cm"].to_numpy(),
            fill_value="extrapolate"
        )
        genomic_intervals = genomic_intervals(self.positions)

        lengths = genomic_intervals[1:] - genomic_intervals[:-1]
        bp = lengths / lengths.sum()

        self.cm_positions = genomic_intervals
        self.breakpoint_probability = bp

    def build_windows(self) -> None:
        """
        Deterministic fixed windows for fitting local models.

        These are NOT the variable-length generation segments.
        """
        if self.cm_positions is None:
            raise ValueError("Run get_chm_info() first.")

        n_snps = len(self.positions)
        self.windows = []

        start = 0
        wid = 0

        while start < n_snps:
            start_cm = self.cm_positions[start]
            end = start + 1

            while end < n_snps and (self.cm_positions[end - 1] - start_cm) < self.window_size_cM:
                end += 1

            # Guardrails on SNP count
            if (end - start) < self.min_window_snps:
                end = min(n_snps, start + self.min_window_snps)

            if (end - start) > self.max_window_snps:
                end = start + self.max_window_snps

            end = min(end, n_snps)
            if end <= start:
                end = min(n_snps, start + 1)

            self.windows.append(
                Window(
                    window_id=wid,
                    start_idx=start,
                    end_idx=end,
                    start_bp=int(self.positions[start]),
                    end_bp=int(self.positions[end - 1]),
                    start_cm=float(self.cm_positions[start]),
                    end_cm=float(self.cm_positions[end - 1]),
                )
            )

            start = end
            wid += 1

    def extract_reference_haplotypes(self) -> None:
        """
        Convert GT array to haplotype matrix of shape (2*n_samples, n_snps).

        Ordering matches laidataset.py:
          first all maternal haplotypes, then all paternal haplotypes.
        """
        if self.gt is None:
            raise ValueError("Run load_reference() first.")

        maternal = self.gt[:, :, 0].T
        paternal = self.gt[:, :, 1].T
        self.haplotypes = np.vstack([maternal, paternal]).astype(np.uint8)

    # ------------------------------------------------------------------
    # Window model fitting
    # ------------------------------------------------------------------

    def _choose_k(self, H: np.ndarray) -> int:
        n_haps = H.shape[0]
        n_unique = np.unique(H, axis=0).shape[0]

        if n_haps <= 1 or n_unique <= 1:
            return 1

        k = int(np.sqrt(max(2, n_haps / 2)))
        k = max(self.k_min, min(k, self.k_max, n_unique))
        return max(1, k)

    def _fit_window_model(self, H: np.ndarray) -> Dict[str, Any]:
        """
        Fit one local KMeans + Bernoulli mixture model.

        H shape:
            (n_haps, n_snps_in_window)
        """
        if H.shape[0] == 0:
            raise ValueError("Cannot fit a window model with zero haplotypes.")

        k = self._choose_k(H)

        if k == 1:
            labels = np.zeros(H.shape[0], dtype=int)
        else:
            km = KMeans(
                n_clusters=k,
                random_state=self.seed,
                n_init=self.n_init,
            )
            labels = km.fit_predict(H)

        weights = []
        probs = []
        cluster_sizes = []

        for c in range(k):
            Hc = H[labels == c]
            if Hc.shape[0] == 0:
                continue

            nk = Hc.shape[0]
            pk = (Hc.sum(axis=0) + self.alpha) / (nk + 2 * self.alpha)

            weights.append(nk)
            probs.append(pk)
            cluster_sizes.append(nk)

        weights = np.array(weights, dtype=float)
        weights /= weights.sum()

        probs = np.vstack(probs).astype(np.float64)
        cluster_sizes = np.array(cluster_sizes, dtype=int)

        return {
            "weights": weights,
            "probs": probs,
            "cluster_sizes": cluster_sizes,
            "k": int(len(weights)),
            "n_haps": int(H.shape[0]),
            "n_snps": int(H.shape[1]),
        }

    def fit_window_models(self) -> None:
        if self.haplotypes is None:
            raise ValueError("Run extract_reference_haplotypes() first.")
        if not self.windows:
            raise ValueError("Run build_windows() first.")

        self.window_models = []
        for win in self.windows:
            H = self.haplotypes[:, win.start_idx:win.end_idx]
            model = self._fit_window_model(H)
            self.window_models.append(model)

    # ------------------------------------------------------------------
    # Segment generation using laidataset.py-style breakpoint process
    # ------------------------------------------------------------------

    def _sample_breakpoints(self, gen: int) -> np.ndarray:
        """
        Use the SAME breakpoint-count logic as laidataset.py admix().

        Returns:
            array like [0, b1, b2, ..., n_snps]
        """
        if self.chm_length_morgans is None or self.breakpoint_probability is None:
            raise ValueError("Run get_chm_info() first.")

        chm_length_snps = len(self.positions)

        # Exact laidataset.py logic
        num_crossovers = int(sum(np.random.poisson(self.chm_length_morgans, size=gen)))

        if num_crossovers == 0:
            return np.array([0, chm_length_snps], dtype=int)

        max_possible = chm_length_snps - 1
        num_crossovers = min(num_crossovers, max_possible)

        breakpoints = np.random.choice(
            np.arange(1, chm_length_snps),
            size=num_crossovers,
            replace=False,
            p=self.breakpoint_probability
        )
        breakpoints = np.sort(breakpoints)
        breakpoints = np.concatenate(([0], breakpoints, [chm_length_snps]))

        return breakpoints.astype(int)

    def _window_for_segment(self, begin: int, end: int) -> int:
        """
        Choose which fitted local window model will govern this segment.

        v1 rule:
          use the window containing the segment midpoint.
        """
        mid = (begin + end - 1) // 2

        for i, win in enumerate(self.windows):
            if win.start_idx <= mid < win.end_idx:
                return i

        return len(self.windows) - 1

    def _sample_segment_from_model(self, model: Dict[str, Any], seg_len: int) -> np.ndarray:
        """
        Sample one segment from one fitted local model.

        If seg_len differs from the fitted window length:
          - truncate if shorter
          - tile repeated draws if longer
        """
        cluster_idx = np.random.choice(len(model["weights"]), p=model["weights"])
        p = model["probs"][cluster_idx]
        model_len = len(p)

        if seg_len == model_len:
            return np.random.binomial(1, p, size=model_len).astype(np.uint8)

        if seg_len < model_len:
            return np.random.binomial(1, p[:seg_len], size=seg_len).astype(np.uint8)

        # seg_len > model_len
        out = np.zeros(seg_len, dtype=np.uint8)
        filled = 0
        while filled < seg_len:
            chunk_len = min(model_len, seg_len - filled)
            out[filled:filled + chunk_len] = np.random.binomial(
                1, p[:chunk_len], size=chunk_len
            ).astype(np.uint8)
            filled += chunk_len
        return out

    def _sample_one_haplotype(self, gen: int) -> np.ndarray:
        """
        Generate one full-length synthetic haplotype:
          - variable-length segments from Poisson/map breakpoints
          - segment content drawn from local KMeans-based models
        """
        n_snps = len(self.positions)
        hap = np.zeros(n_snps, dtype=np.uint8)

        breakpoints = self._sample_breakpoints(gen)

        for i in range(len(breakpoints) - 1):
            begin = breakpoints[i]
            end = breakpoints[i + 1]
            seg_len = end - begin

            win_idx = self._window_for_segment(begin, end)
            model = self.window_models[win_idx]

            hap[begin:end] = self._sample_segment_from_model(model, seg_len)

        return hap

    def sample_haplotypes(self, n_haplotypes: int, gen: Optional[int] = None) -> np.ndarray:
        """
        Generate full synthetic haplotypes.

        Output shape:
            (n_haplotypes, n_snps)
        """
        if not self.window_models:
            raise ValueError("Run fit_window_models() first.")

        if gen is None:
            gen = self.synthetic_gen

        out = np.zeros((n_haplotypes, len(self.positions)), dtype=np.uint8)
        for i in range(n_haplotypes):
            out[i, :] = self._sample_one_haplotype(gen=gen)
        return out

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _scaffold_hash(self) -> str:
        if self.positions is None or self.refs is None or self.alts is None:
            raise ValueError("Scaffold not loaded.")

        payload = pd.DataFrame({
            "chrom": [self.chm] * len(self.positions),
            "pos": self.positions.astype(str),
            "ref": self.refs.astype(str),
            "alt": self.alts.astype(str),
        }).to_csv(sep="\t", index=False)

        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def write_windows(self, out_file: str) -> None:
        if not self.windows:
            raise ValueError("No windows to write.")
        pd.DataFrame([asdict(w) for w in self.windows]).to_csv(out_file, sep="\t", index=False)

    def write_output(
        self,
        out_prefix: str,
        synthetic_haplotypes: np.ndarray,
        write_windows: bool = True,
    ) -> None:
        if os.path.dirname(out_prefix):
            os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

        np.savez_compressed(
            out_prefix + ".npz",
            haplotypes=synthetic_haplotypes.astype(np.uint8),
        )

        meta = {
            "chromosome": self.chm,
            "ancestry_label": self.ancestry_label,
            "n_snps": int(len(self.positions)),
            "n_haplotypes": int(synthetic_haplotypes.shape[0]),
            "window_size_cM": float(self.window_size_cM),
            "synthetic_gen": int(self.synthetic_gen),
            "min_window_snps": int(self.min_window_snps),
            "max_window_snps": int(self.max_window_snps),
            "seed": int(self.seed),
            "alpha": float(self.alpha),
            "k_min": int(self.k_min),
            "k_max": int(self.k_max),
            "n_windows": int(len(self.windows)),
            "reference_vcf": self.reference_vcf,
            "genetic_map_file": self.genetic_map_file,
            "snp_manifest_file": self.snp_manifest_file,
            "scaffold_hash": self._scaffold_hash(),
            "first_position": int(self.positions[0]),
            "last_position": int(self.positions[-1]),
        }

        with open(out_prefix + ".json", "w") as f:
            json.dump(meta, f, indent=2)

        if write_windows:
            self.write_windows(out_prefix + ".windows.tsv")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        n_haplotypes: int,
        out_prefix: str,
        gen: Optional[int] = None,
        write_windows: bool = True,
    ) -> None:
        self.load_reference()
        self.get_chm_info()
        self.build_windows()
        self.extract_reference_haplotypes()
        self.fit_window_models()
        synthetic = self.sample_haplotypes(n_haplotypes=n_haplotypes, gen=gen)
        self.write_output(out_prefix, synthetic, write_windows=write_windows)
