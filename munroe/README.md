# XKCD Cleaned Dataset #

Dataset used in ACL 2018 paper "'Lighter' Can Still be Dark: Modeling Comparative Color Words".

Source data available from the 'Supplement' link at [Brian McMahan's page](http://mcmahan.io/lux/)


## Contact ##

**Authors**: Olivia Winn, Smaranda Muresan

**Contact**: <olivia@cs.columbia.edu>




## Dataset ##

The data is compressed in [xkcd_colordata.tar.gz](xkcd_colordata.tar.gz]). Training, development, and testing data are separated by file extension (.train, .dev, and .test respectively); all are binary files where each line is an RGB datapoint with the channel values separated by whitespace. The channel values are 0-255, and the files can be read using pickle.

*Ex: when converted to text format:*
> 250 30 180




## Files ##

**[words\_to\_labels.txt](words_to_labels.txt)**

Each line contains the words of each color label and the corresponding file label, with the two separated by a comma and the words separated by whitespace. Used for transforming the file labels to their corresponding words.

*Ex:*
> light gray blue,lightgrayblue




**[comparatives.txt](comparatives.txt)**

**[quantifiers.txt](quantifiers.txt)**

Each line is a quantifying label word followed by its comparative version, separated by a comma. The two files contain all words which were treated as quantifiers of the subsequent words in a color label if they were the first word in the label. [comparatives.txt](comparatives.txt) contains words with comparative forms, [quantifiers.txt](quantifiers.txt) those without.

For example, 'light' is in the [comparatives.txt](comparatives.txt) list as 'light,lighter', so 'light gray blue' can be transformed to 'lighter' + 'gray blue'.