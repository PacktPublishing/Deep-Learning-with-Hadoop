import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.net.URL;


public class RNN {

    
    public static final String DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";
  
    public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_w2vSentiment/");
   
    public static final String WORD_VECTORS_PATH = "/PATH/TO/YOUR/VECTORS/GoogleNews-vectors-negative300.bin";


    public static void main(String[] args) throws Exception {
  
        downloadData();

        int batchSize = 50;     
        int vectorSize = 300;    
        int nEpochs = 5;         
        int truncateReviewsToLength = 300;   

         
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .updater(Updater.RMSPROP)
                .regularization(true).l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
                .learningRate(0.0018)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(200)
                        .activation("softsign").build())
                .layer(1, new RnnOutputLayer.Builder().activation("softmax")
                        .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(200).nOut(2).build())
                .pretrain(false)
				.backprop(true)
				.build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));
 
        WordVectors wordVectors = WordVectorSerializer.loadGoogleModel(new File(WORD_VECTORS_PATH), true, false);
        DataSetIterator train = new AsyncDataSetIterator(new SentimentExampleIterator(DATA_PATH,wordVectors,batchSize,truncateReviewsToLength,true),1);
        DataSetIterator test = new AsyncDataSetIterator(new SentimentExampleIterator(DATA_PATH,wordVectors,100,truncateReviewsToLength,false),1);

        System.out.println("Starting training");
        for( int i=0; i<nEpochs; i++ ){
            net.fit(train);
            train.reset();
            System.out.println("Epoch " + i + " complete. Starting evaluation:");

            
            Evaluation evaluation = new Evaluation();
            while(test.hasNext()){
                DataSet t = test.next();
                INDArray features = t.getFeatureMatrix();
                INDArray lables = t.getLabels();
                INDArray inMask = t.getFeaturesMaskArray();
                INDArray outMask = t.getLabelsMaskArray();
                INDArray predicted = net.output(features,false,inMask,outMask);

                evaluation.evalTimeSeries(lables,predicted,outMask);
            }
            test.reset();

            System.out.println(evaluation.stats());
        }


    
    }

    private static void downloadData() throws Exception {
         
        File directory = new File(DATA_PATH);
        if(!directory.exists()) directory.mkdir();

        
        String archizePath = DATA_PATH + "aclImdb_v1.tar.gz";
        File archiveFile = new File(archizePath);
        String extractedPath = DATA_PATH + "aclImdb";
        File extractedFile = new File(extractedPath);

        if( !archiveFile.exists() ){
            System.out.println("Starting data download (80MB)...");
            FileUtils.copyURLToFile(new URL(DATA_URL), archiveFile);
            System.out.println("Data (.tar.gz file) downloaded to " + archiveFile.getAbsolutePath());
            //Extract tar.gz file to output directory
            extractTarGz(archizePath, DATA_PATH);
        } else {
            //Assume if archive (.tar.gz) exists, then data has already been extracted
            System.out.println("Data (.tar.gz file) already exists at " + archiveFile.getAbsolutePath());
            if( !extractedFile.exists()){
            	//Extract tar.gz file to output directory
            	extractTarGz(archizePath, DATA_PATH);
            } else {
            	System.out.println("Data (extracted) already exists at " + extractedFile.getAbsolutePath());
            }
        }
    }

    private static final int BUFFER_SIZE = 4096;
    private static void extractTarGz(String filePath, String outputPath) throws IOException {
        int fileCount = 0;
        int dirCount = 0;
        System.out.print("Extracting files");
        try(TarArchiveInputStream tais = new TarArchiveInputStream(
                new GzipCompressorInputStream( new BufferedInputStream( new FileInputStream(filePath))))){
            TarArchiveEntry entry;

             
            while ((entry = (TarArchiveEntry) tais.getNextEntry()) != null) {
                 
                if (entry.isDirectory()) {
                    new File(outputPath + entry.getName()).mkdirs();
                    dirCount++;
                }else {
                    int count;
                    byte data[] = new byte[BUFFER_SIZE];

                    FileOutputStream fos = new FileOutputStream(outputPath + entry.getName());
                    BufferedOutputStream dest = new BufferedOutputStream(fos,BUFFER_SIZE);
                    while ((count = tais.read(data, 0, BUFFER_SIZE)) != -1) {
                        dest.write(data, 0, count);
                    }
                    dest.close();
                    fileCount++;
                }
                if(fileCount % 1000 == 0) System.out.print(".");
            }
        }

        System.out.println("\n" + fileCount + " files and " + dirCount + " directories extracted to: " + outputPath);
    }
}
