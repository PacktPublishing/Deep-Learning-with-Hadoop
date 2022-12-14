
### Get this product for $5

<i>Packt is having its biggest sale of the year. Get this eBook or any other book, video, or course that you like just for $5 each</i>


<b><p align='center'>[Buy now](https://packt.link/9781787124769)</p></b>


<b><p align='center'>[Buy similar titles for just $5](https://subscription.packtpub.com/search)</p></b>


# Deep Learning with Hadoop
This is the code repository for [Deep Learning with Hadoop](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-hadoop?utm_source=github&utm_medium=repository&utm_campaign=9781787124769), published by Packt. It contains all the supporting project files necessary to work through the book from start to finish.


## About the Book
Deep Learning involves extracting features and insights from multiple layers of the data. When applied to the world of Hadoop you have the potential to get even more from your data than before. This book will teach you how to deploy the deep learning networks with Hadoop for optimal performance.

Starting with understanding what deep learning is and what the various models associated with deep learning are, this book will then show you how to set up the Hadoop environment for deep learning. In this book, you will also learn how to overcome the challenges that you face while implementing distributed deep learning with Hadoop. The book will also show you how you can implement and parallelize Deep Belief Networks, CNN, RNN, RBM, and much more using the popular deep learning library deeplearning4j.

Get in-depth mathematical explanations, visual representations to understand the implementation of Denoising AutoEncoders with deeplearning4j. To give you a more practical perspective, the book will also teach you how you can implement image classification, audio processing and natural language processing on Hadoop.

By the end of this book, you will know how to deploy deep learning in distributed systems using Hadoop

## Instructions and Navigation
All of the code is organized into folders. For example, Chapter02.



The code will look like the following:

        int nEpochs = 30;
        int numInputs = 2;
        int numOutputs = 2;
        int numHiddenNodes = 20;

        
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File("saturn_data_train.csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,0,2);

        
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File("saturn_data_eval.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,0,2);

      
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false)
				.backprop(true)
				.build();



## Note:
Chapter 01 and Chapter 07 does not contain code files.

## Related Hadoop Products
* [Hadoop: Data Processing and Modelling](https://www.packtpub.com/big-data-and-business-intelligence/hadoop-data-processing-and-modelling?utm_source=github&utm_medium=repository&utm_content=9781787125162)

* [Hadoop Blueprints](https://www.packtpub.com/big-data-and-business-intelligence/hadoop-blueprints?utm_source=github&utm_medium=repository&utm_content=9781783980307)

* [Elasticsearch for Hadoop](https://www.packtpub.com/big-data-and-business-intelligence/elasticsearch-hadoop?utm_source=github&utm_medium=repository&utm_content=9781785288999)


### Suggestions and Feedback
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSe5qwunkGf6PUvzPirPDtuy1Du5Rlzew23UBp2S-P3wB-GcwQ/viewform) if you have any feedback or suggestions.
### Download a free PDF

 <i>If you have already purchased a print or Kindle version of this book, you can get a DRM-free PDF version at no cost.<br>Simply click on the link to claim your free PDF.</i>
<p align="center"> <a href="https://packt.link/free-ebook/9781787124769">https://packt.link/free-ebook/9781787124769 </a> </p>