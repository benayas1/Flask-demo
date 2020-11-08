Image Dataset

Images:

Our image dataset contains a total of 64,832 advertisement images verified by human annotators on Amazon Mechanical Turk. Images are split into 11 folders (subfolder-0 to subfolder-10).
To obtain the dataset, please just download all compressed zip files and extract them all into the same folder.

https://storage.googleapis.com/ads-dataset/subfolder-0.zip
https://storage.googleapis.com/ads-dataset/subfolder-1.zip
https://storage.googleapis.com/ads-dataset/subfolder-2.zip
https://storage.googleapis.com/ads-dataset/subfolder-3.zip
https://storage.googleapis.com/ads-dataset/subfolder-4.zip
https://storage.googleapis.com/ads-dataset/subfolder-5.zip
https://storage.googleapis.com/ads-dataset/subfolder-6.zip
https://storage.googleapis.com/ads-dataset/subfolder-7.zip
https://storage.googleapis.com/ads-dataset/subfolder-8.zip
https://storage.googleapis.com/ads-dataset/subfolder-9.zip
https://storage.googleapis.com/ads-dataset/subfolder-10.zip

As explained in the paper, we first use a ResNet classifier to automatically classify candidate advertisements into "ads" or "not ads". (This is followed by a human verification task.) For this automatic classification of images as "ads" or "not ads", we use subfolder-10 as positive samples for training the ResNet, while the negative samples are provided as "resnet_negative.zip". If you just want to use the dataset, there is NO NEED to download these non-ads.

https://storage.googleapis.com/ads-dataset/resnet_negative.zip

Annotation Files:

The annotation files are in json format.

The key in the file is the relative path ({subfolder}/{image_name})
The value is a list of annotations from different annotators. For each image, we posed the same question to 3-5 different annotators. We show all obtained annotations in this dataset.

For Q/A ("Action", "Reason") and "Slogans", we provide unprocessed annotations as free-form sentences. For example:
{
  ...,
  "7/62717.jpg": ["Because they are a patriotic company.",
                  "Because it is a fun, American company; a staple.",
                  "Because it is the American spirit."],
  ...
}

The file QA_Combined_Action_Reason.json contains results for the study in which the annotators gave a single combined answer to the "What should I do" and "Why should I do it" questions.
For the study in which the question was split into separate "What should I do" and "Why should I do it" questions, the file QA_Action.json contains the results for the "What" questions, and the file QA_Reason.json contains the results for the "Why" questions for the same images. 

For "Topics", "Sentiments", and "Strategies", we show the class ID in the annotation files, and a corresponding mapping is provided in a separate txt file (Topics_List.txt, Sentiments_List.txt, and Strategies_List, respectively). The class ID identifies the particular topic, sentiment, or strategy that the annotator selected. 

For example, for topics, if the value in the key-value pair is a list ["10", "11", "10"], this means that two annotators selected topic number 10 ("Electronics") and one selected topic number 11 ("Phone"). 
If the list contains strings, then some annotator(s) selected the "Other" option, and the string is the text that the annotator(s) entered after selecting the "Other" option. For example:
{
  ...,
  '9/91899.jpg': ['10', '10', 'musical instruments'],
  ...
}
 
Note that for "Sentiments" and "Strategies", one annotator could select multiple categories, so the value in the key-value pair is a list of lists. For example:
{
  ...,
   '3/40803.jpg': [['18', '19'],
                   ['19'],
                   ['18', '19', '25']],
  ...
}

For "Symbols", the value in the key-value pair is a list of annotations from multiple annotators, and each annotation is itself a list. The first 4 elements of that list are the coordinates of the bounding box that the annotator drew, and the last element is the phrase that annotator provided as a symbol label for that box (i.e. the "signified", using terminology from our paper). The coordinates x, y range from 0 to 500 as each image was scaled to 501x501. For example:
{
  ...,
   u'9/86039.jpg': [[356.0, 318.0, 230.0, 198.0, 'Dangerous'],
                    [212.0, 200.0, 417.0, 376.0, 'wild/adventurous'],
                    [224.0, 193.0, 446.0, 436.0, 'Adventure'],
                    [252.0, 236.0, 376.0, 335.0, 'Risk/Thrill'],
                    [259.0, 224.0, 387.0, 372.0, 'skydiving']],
  ...
}

Citation:

If you use our data, please cite the following paper:

Automatic Understanding of Image and Video Advertisements. Zaeem Hussain, Mingda Zhang, Xiaozhong Zhang, Keren Ye, Christopher Thomas, Zuha Agha, Nathan Ong, Adriana Kovashka. To appear, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.
