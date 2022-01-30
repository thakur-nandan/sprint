For reproducing the reported results, please go with the `ndigits-round` (i.e. `round(term weight * 100)`) quantization. Specifically, one can run the scripts in this order:

0.download-distilsplade_max.sh

1.encode.beir_scifact-distilsplade_max-float.sh

2.quantize.beir_scifact-distilsplade_max-2digits.sh  (**Note here it will do the quantization**)

3.index.beir_scifact-distilsplade_max-2digits.sh 

4.reformat_query.beir_scifact.sh

5.search.beir_scifact-distilsplade_max-2digits.sh

6.evaluate.beir_scifact-distilsplade_max-2digits.sh