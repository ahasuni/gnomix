### When Using Pre-Trained Models
gnomix.py loads and uses a pre-trained Gnomix model to predict the ancestry for a given *<query_file>* and a chromosome.

To execute the program with a pre-trained model run:
```
$ python3 gnomix.py <query_file> <output_folder> <chr_nr> <phase> <path_to_model> 
```

where 
- <*query_file*> is a .vcf or .vcf.gz file containing the query haplotypes which are to be analyzed (see example in the **demo/data/** folder)
- <*output_folder*> is where the results will be written (see details in **Output** below and an example in the **demo/data/** folder)
- <*chr_nr*> is the chromosome number
- <*phase*> is either True or False corresponding to the intent of using the predicted ancestry for phasing correction (see details in **Phasing** below and in the **gnofix/** folder). Note that initial phasing (using a program like beagle, shapeit, or eagle) must still have been performed first.
- <*path_to_model*> is a path to the model used for predictions (see **Pre-trained Models** below)

### Downloading pre-trained models
In order to incorporate our pre-trained models into your pipeline, please use the following command to download pre-trained models for the whole human genome. The SNPs used for our pre-trained models are also included in the form of a plink .bim file for every chromosome.
```
sh download_pretrained_models.sh
```
This creates a folder called **pretrained_gnomix_models**. For each chromosome, we publish a *default_model.pkl* which can be used as a pre-trained model in the <*path_to_model*> field and a *.bim* file as explained above.

When making predictions, the input to the model is an intersection of the pre-trained model SNP positions and the SNP positions from the <query_file>. That means that the set of positions that are only in the original training input used to create the model (and not in the query samples) are encoded as missing, while the set of positions only in the <query_file> are discarded. We suggest that you attempt to have your query samples include as many model snps (listed in the *.bim* files) as possible (never less than 80% for sufficient accuracy. When the script is executed, it will log the intersection-ratio between these model snps and the snps in your query samples, since the anceestry inference performance will depend on how many of the model's snp positions are missing in your query samples. *If the intersection is low, you must either impute your query samples to match the full set of snps that are present in the pre-trained model or you must train your own new model using references that contain all the snps in your query samples.* N.B. Your query samples must have snps that are defined on the same strand as in the model. You can use the included model *.bim* files as a reference to find and then flip any snps in your query samples that are defined on the opposite strand. (If this step is not performed your query samples will appear to have snps containing variation unseen during the model's training and will thus be be assigned unpredictable ancestries.)

The models named **default_model.pkl** are trained on hg build 37 references from the following biogeographic regions: *Subsaharan African (AFR), East Asian (EAS), European (EUR), Native American (NAT), Oceanian (OCE), South Asian (SAS), and West Asian (WAS)* and the model labels and predicts them as 0, 1, .., 6 respectively. The populations used to train these ancestries are given in the supplementary section of the reference provided at the bottom of this readme.

### Demo

After downloading our pre-trained models, one can demo the software in inference mode by running:
```
python3 gnomix.py demo/data/small_query_chr22.vcf.gz demo_output 22 True pretrained_gnomix_models/chr22/model_chm_22.pkl
```
This small query file contains only 9 samples of European, East Asian and African ancestry. The execution should take around a minute on a standard laptop. The inference can be analyzed, for example in the file demo_output/quer_results.msp, where we expect to see those three ancestries being inferred. For more details on those analysis, see the section on output below.

For more demos with training and larger datasets, see the [demo](demo.ipynb) notebook *demo.ipynb*.
