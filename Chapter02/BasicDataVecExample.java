import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.CategoricalColumnCondition;
import org.datavec.api.transform.condition.column.DoubleColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.condition.ConditionalReplaceValueTransform;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.transform.transform.time.StringToTimeTransform;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.datavec.spark.functions.RecordReaderFunction;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;


public class BasicDataVecExample {

    public static  void main(String[] args) throws Exception {

       
        Schema inputDataSchema = new Schema.Builder()
            
            .addColumnString("DateTimeString")
            
            .addColumnsString("CustomerID", "MerchantID")
          
            .addColumnInteger("NumItemsInTransaction")
            .addColumnCategorical("MerchantCountryCode", Arrays.asList("USA","CAN","FR","MX"))
           
            .addColumnDouble("TransactionAmountUSD",0.0,null,false,false)  
            .addColumnCategorical("FraudLabel", Arrays.asList("Fraud","Legit"))
            .build();

       
        System.out.println("Input data schema details:");
        System.out.println(inputDataSchema);

        System.out.println("\n\nOther information obtainable from schema:");
        System.out.println("Number of columns: " + inputDataSchema.numColumns());
        System.out.println("Column names: " + inputDataSchema.getColumnNames());
        System.out.println("Column types: " + inputDataSchema.getColumnTypes());


      

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
            
            .removeColumns("CustomerID","MerchantID")

          
            .filter(new ConditionFilter(
                new CategoricalColumnCondition("MerchantCountryCode", ConditionOp.NotInSet, new HashSet<>(Arrays.asList("USA","MX")))))

           
            .conditionalReplaceValueTransform(
                "TransactionAmountUSD",      
                new DoubleWritable(0.0),    
                new DoubleColumnCondition("TransactionAmountUSD",ConditionOp.LessThan, 0.0)) 

           
            .stringToTimeTransform("DateTimeString","YYYY-MM-DD HH:mm:ss.SSS", DateTimeZone.UTC)

         
            .renameColumn("DateTimeString", "DateTime")

            
            .transform(new DeriveColumnsFromTimeTransform.Builder("DateTime")
                .addIntegerDerivedColumn("HourOfDay", DateTimeFieldType.hourOfDay())
                .build())

      
            .removeColumns("DateTime")
 
            .build();


         
        Schema outputSchema = tp.getFinalSchema();

        System.out.println("\n\n\nSchema after transforming data:");
        System.out.println(outputSchema);


        

        SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");
        conf.setAppName("DataVec Example");

        JavaSparkContext sc = new JavaSparkContext(conf);
		//You can defined the directory of the path according to your convinience
		
        String directory = new ClassPathResource("exampledata.csv").getFile().getParent(); 
        JavaRDD<String> stringData = sc.textFile(directory);

        
        RecordReader rr = new CSVRecordReader();
        JavaRDD<List<Writable>> parsedInputData = stringData.map(new StringToWritablesFunction(rr));


        SparkTransformExecutor exec = new SparkTransformExecutor();
        JavaRDD<List<Writable>> processedData = exec.execute(parsedInputData, tp);

   
        JavaRDD<String> processedAsString = processedData.map(new WritablesToStringFunction(","));
        //processedAsString.saveAsTextFile("file://your/local/save/path/here");   //To save locally
        //processedAsString.saveAsTextFile("hdfs://your/hdfs/save/path/here");   //To save to hdfs

        List<String> processedCollected = processedAsString.collect();
        List<String> inputDataCollected = stringData.collect();


        System.out.println("\n\n---- Original Data ----");
        for(String s : inputDataCollected) System.out.println(s);

        System.out.println("\n\n---- Processed Data ----");
        for(String s : processedCollected) System.out.println(s);


        System.out.println("\n\nDONE");
    }

}
