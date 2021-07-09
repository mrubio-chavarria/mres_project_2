
reads_folder=$1
basecalls_folder=$2
reference=$3

# Convert from multi to single format
echo "Convert from multi to single format"
multi_to_single_fast5 --input_path $reads_folder/multi --save_path $reads_folder/single
echo "Conversion completed"

# Flatten the folder structure
echo "Flattening folder structure:"
rm $reads_folder/single/*_mapping.txt
for folder in $reads_folder/single/*
do
  for file in $folder/*
    do
      cp $file $reads_folder/single/"${file##*/}"
    done
  echo "Folder ${folder##*/} flattened"
  rm -r $folder
done
echo "Flattening finished"

# Annotate the basecalled seuences
# NOTE: the algorithm processes all the reads, but only writes those in the fastq file
# that is the reason why the percentages are different in the terminal
echo "Annotating fast5 files:"
for file in $basecalls_folder/*.fastq
do
  # Annotate the single-formated reads
  echo "BASECALLS ACCESSED: ${file##*/}"
  tombo preprocess annotate_raw_with_fastqs --fast5-basedir $reads_folder/single --fastq-filenames $file --overwrite
done
echo "Annotation finished"

# Resquiggle the fast files
tombo resquiggle $reads_folder/single $reference --processes 4 --num-most-common-errors 5 --overwrite

# Filter reads
# Matching score should be between 0.5 and 3, the bigger the less strict
score=0.5
tombo filter raw_signal_matching --fast5-basedirs $reads_folder/single --signal-matching-score $score
