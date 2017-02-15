import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.Arrays;
import java.util.Collections;


public class StackedAutoEncoder{

    private static Logger log = LoggerFactory.getLogger(StackedAutoEncoder.class);

    public static void main(String[] args) throws Exception {
       // final int numRows = 28;
       // final int numColumns = 28;
        int outputNum = 10;
		int inputNum = 1000;
        int numSamples = 60000;
        int batchSize = 1024;
        int iterations = 10;
        int seed = 123;
        int listenerFreq = batchSize / 5;

        log.info("Load data....");
        DataSetIterator iter = new MnistDataSetIterator(batchSize,numSamples,true);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
           .seed(seed)
           .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
           .gradientNormalizationThreshold(1.0)
           .iterations(iterations)
           .momentum(0.5)
           .momentumAfter(Collections.singletonMap(3, 0.9))
           .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
           .list()
		   .layer(0, new AutoEncoder.Builder()
				   .nIn(inputNum)
				   .nOut(500)
                   .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                   .corruptionLevel(0.3)
                   .build())
           .layer(1, new AutoEncoder.Builder()
		                .nIn(500)
						.nOut(250)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                        .corruptionLevel(0.3)
                        .build())
           .layer(2, new AutoEncoder.Builder()
		                .nIn(250)
						.nOut(125)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                        .corruptionLevel(0.3)
                        .build())
		   .layer(3, new AutoEncoder.Builder()
		                .nIn(125)
						.nOut(50)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                        .corruptionLevel(0.3)
                        .build())
           .layer(4, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
		                .activation("softmax")
                        .nIn(75)
						.nOut(outputNum)
						.build())
           .pretrain(true)
		   .backprop(false)
           .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        log.info("Train model....");
        model.fit(iter); 

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);

        DataSetIterator testIter = new MnistDataSetIterator(100,10000);
        while(testIter.hasNext()) {
            DataSet testMnist = testIter.next();
            INDArray predict2 = model.output(testMnist.getFeatureMatrix());
            eval.eval(testMnist.getLabels(), predict2);
        }

        log.info(eval.stats());

    }

}