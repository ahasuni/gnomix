# Configuration file templates

In this folder we provide different configuration file templates corresponding to different use cases. Some are critical for execution, while others are mainly intended to improve results. If you find yourself tweaking a config file for a particular use case, consider sharing it (for example, by creating an issue) so other researchers can benefit from it too :)

## config.yaml

This is the default, compromise configuration for all LAI analyses on the human genome, and it is the same as the one in the root folder. It is designed for generic whole-genome data, but it also works well with array data.

## config_array.yaml

This is an optimization for array data on the human genome. For array data, variations of the string kernels tend to perform significantly better than the default logistic regression model. Since array data file sizes are also smaller, using more complex models is not only useful for performance for array, since they have fewer snp features, but also typically has less time and memory downside, as opposed to whole genome, for the same reason.

The only difference is the base models that are used:

- *model.inference = best*

## config_tracts.yaml

This config is intended to prioritize tract length accuracy for population analyses, especially when results will be used with Tracts and similar downstream tools. In these settings, ancestry segment length is important, since the length distribution will be used, so allowing rapid switching of local ancestry predictions to maximize site-specific ancestry accuracy in the face of uncertainty between two ancestries, is undesired. Instead, a best choice selection for the entire tract is desired, so that its length is not split into many small fragments. Use this template when your objective is maximizing tract-level accuracy for downstream population demographic analyses, rather than maximizing site-specific ancestry accuracy.

## config_lowdata.yaml

When training with a small sample size, for example fewer than 20 samples or fewer than 5 samples per population, one can barely afford to hold out data for validation. That is why we recommend not using a validation split in this setting:

- *simulation.split.ratios.val = 0*

Given the small data size, we also recommend using the best models independently from the data type:

- *model.inference = best*

If computation time is too high, the default (empty) setting is better.

## config_plants.yaml

This config has been used for analyzing date palm data. It extends the *lowdata* config file with an additional change in the window size. Since that specific dataset had a much higher density of SNPs per centiMorgan, we reduce the model window size:

- *window_size_cM = 0.05* (smaller)
- *smooth_size = 25* (smaller)
