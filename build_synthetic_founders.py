import argparse
import yaml

from src.synthetic_founders import SyntheticFounderGenerator


def main():
    parser = argparse.ArgumentParser(description="Build synthetic ancestry-specific haplotypes from a phased VCF.")
    parser.add_argument("--vcf", required=True, help="Input phased VCF/VCF.GZ of unadmixed individuals")
    parser.add_argument("--genetic-map", required=True, help="Genetic map file")
    parser.add_argument("--snp-manifest", required=True, help="Shared SNP scaffold manifest TSV")
    parser.add_argument("--ancestry", required=True, help="Ancestry label for this contributor panel")
    parser.add_argument("--chrom", required=True, help="Chromosome")
    parser.add_argument("--config", required=True, help="Gnomix config.yaml")
    parser.add_argument("--n-haplotypes", required=True, type=int, help="Number of synthetic haplotypes to generate")
    parser.add_argument("--out-prefix", required=True, help="Output prefix for .npz/.json")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument("--gen", type=int, default=None, help="Override synthetic generation parameter")
    parser.add_argument("--no-write-windows", action="store_true", help="Do not write windows TSV")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)

    gen = SyntheticFounderGenerator(
        chm=args.chrom,
        reference_vcf=args.vcf,
        genetic_map_file=args.genetic_map,
        ancestry_label=args.ancestry,
        snp_manifest_file=args.snp_manifest,
        config=config,
        seed=args.seed,
    )

    gen.run(
        n_haplotypes=args.n_haplotypes,
        out_prefix=args.out_prefix,
        gen=args.gen,
        write_windows=(not args.no_write_windows),
    )


if __name__ == "__main__":
    main()
