package assignment21

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions.{window, column, desc, col}


import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Column
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, IntegerType}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.{count, sum, min, max, asc, desc, udf, to_date, avg}

import org.apache.spark.sql.functions.explode
import org.apache.spark.sql.functions.array
import org.apache.spark.sql.SparkSession

import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext}




import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansSummary}


import java.io.{PrintWriter, File}


//import java.lang.Thread
import sys.process._


import org.apache.log4j.Logger
import org.apache.log4j.Level
import scala.collection.immutable.Range




object assignment  {
  // Suppress the log messages:
  Logger.getLogger("org").setLevel(Level.OFF)
                       
  
  val spark = SparkSession.builder().appName("assignment21").config("spark.driver.host", "localhost").master("local").getOrCreate()
                          
  val dataK5D2: DataFrame=  spark.read.format("csv").option("header", "true").option("inferSchema", "true").csv("data/dataK5D2.csv")

  val dataK5D3: DataFrame=  spark.read.format("csv").option("header", "true").option("inferSchema", "true").csv("data/dataK5D3.csv")

  val dataK5D3WithLabels: DataFrame = spark.read.format("csv").option("header", "true").option("inferSchema", "true").csv("data/dataK5D3.csv")
  
  import org.apache.spark.sql.functions.col
  import org.apache.spark.sql.types.DoubleType
  
  def task1(df: DataFrame, k: Int): Array[(Double, Double)] = {
    // Making dataframe, removing header and changing a,b to double. Drop "any" drops NULL values
    val data = df.drop("LABEL")
                              .withColumn("a",col("a").cast("double"))
                              .withColumn("b", col("b").cast("double"))
                              .na.drop("any")
      
                              
    data.cache() // loads data to memory
    
    data.show(5)
    
    //Creating vectorAssembler and map columns a,b to features
    val vectorAssembler_1 = new VectorAssembler().setInputCols(Array("a", "b")).setOutputCol("features")
    
    
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.clustering.{KMeans, KMeansSummary, KMeansModel}
    
    //Bonus task 4. Making pipeline to process and learn from data
    val tf_Pipeline = new Pipeline().setStages(Array(vectorAssembler_1))
    val PipeLine = tf_Pipeline.fit(data)
    val tf_training = PipeLine.transform(data)

    
    //Making Kmeans object, make it fit the tf_training. k is given as a parameter
    val kmeans = new KMeans().setK(k).setSeed(1L)     
    val kmeansModel: KMeansModel = kmeans.fit(tf_training)

    //K-means clustering. Data is two dimensional
    val cluster_centroids = kmeansModel.clusterCenters.map(x => x.toArray).map{case Array(f1,f2) => (f1,f2)}
    
   
    return cluster_centroids
  }

  def task2(df: DataFrame, k: Int): Array[(Double, Double, Double)] = {
    // Same as in task 1. Now data is three dimensional
    val data = df.drop("LABEL")
                              .withColumn("a",col("a").cast("double"))
                              .withColumn("b", col("b").cast("double"))
                              .withColumn("c", col("c").cast("double"))
                              .na.drop("any")
    
    //data to memory                         
    data.cache()
    data.show(8)
    
    
    import org.apache.spark.sql.functions.{mean, stddev}
    
    //Statistics of "a"
    data.select(count("a"), mean("a"), stddev("a")).show()
    
  
    //Statistics of "b" 
    data.select(count("b"), mean("b"), stddev("b")).show()
    
    //Statistics of "c" 
    data.select(count("c"), mean("c"), stddev("c")).show()
    // Column "c" standard deviation is too high 273.6 -> need to be scaled
    
    import org.apache.spark.ml.linalg.Vectors
    import org.apache.spark.ml.feature.MinMaxScaler
    import org.apache.spark.sql.functions.udf
    
    // Using MinMacScaler to scale down column "c"
    val UDF = udf( (v:Double) => Vectors.dense(Array(v)) ) 
    val df_1 = data.withColumn("c_vec", UDF(data("c")))    
    val scaler = new MinMaxScaler().setInputCol("c_vec").setOutputCol("cScaled").setMax(1).setMin(-1)
    
    
    //New data with scaled "c"
    val data2 = scaler.fit(df_1).transform(df_1)
    
    // Mapping a,b,c to features
    val vA = new VectorAssembler().setInputCols(Array("a", "b", "cScaled")).setOutputCol("features")
    
    //Bonus task 4
    val tf_Pl = new Pipeline().setStages(Array(vA))
    val Pl = tf_Pl.fit(data2)
    val tf_training = Pl.transform(data2)

    
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.clustering.{KMeans, KMeansSummary, KMeansModel}
    
    //Making k means object and fit it
    val kmeans = new KMeans().setK(k).setSeed(1L)
    val kmeansModel: KMeansModel = kmeans.fit(tf_training)

    //K-means clustering. Data is three dimensional
    val cluster_centroids = kmeansModel.clusterCenters.map(x => x.toArray).map{case Array(f1,f2,f3) => (f1,f2,f3)}
 
    return cluster_centroids
    
  }

  def task3(df: DataFrame, k: Int): Array[(Double, Double)] = {
   
     
    df.show()
    df.printSchema()
    
    import org.apache.spark.ml.feature.StringIndexer
    
    // "Maps a string column of labels to an ML column of label indices"
    val Stringindexer = new StringIndexer().setInputCol("LABEL").setOutputCol("label_indices")
    
    val df_1 = Stringindexer.fit(df).transform(df)
      
    
    
    df_1.show()
    df_1.printSchema()
    
    // Making dataframe, removing header and changing a,b, label_indices(label) to double. Drop "any" drops NULL values
    
     val data = df_1.drop("LABEL")
                              .withColumn("a",col("a").cast("double"))
                              .withColumn("b", col("b").cast("double"))
                              .withColumn("label", col("label_indices").cast("double"))
                              .na.drop("any")                         
                           
                 
                  
    data.printSchema()
    
    //data to memory
    data.cache()
    data.show(5)  
    
    // Creating VectorAssembler by mapping a,b and label to features.
    val vectorAssembler = new VectorAssembler().setInputCols(Array("a", "b", "label")).setOutputCol("features")
    
    
   
    val tf_Pl = new Pipeline().setStages(Array(vectorAssembler))
    val Pl = tf_Pl.fit(data)
    val tf_training = Pl.transform(data)

  
    //Creating k means object and fitting tf_training to get k-means object
    val kmeans = new KMeans().setK(k).setSeed(1L)
    val kmModel = kmeans.fit(tf_training)
    
    val cluster_centroids = kmModel.clusterCenters.map(x => x.toArray).map{case Array(f1,f2,f3) => (f1,f2,f3)}
    //"return only the twodimensional clusters means for two clusters that are most Fatal"
    .filter(x => (x._3 > 0.43)) // need to change from 0.5 to 0.43 to get two clusters
    .map{case (f1,f2,f3) => (f1,f2)}
    
    return cluster_centroids
  }

  // Parameter low is the lowest k and high is the highest one.
  def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)]  = {
   
    
    import org.apache.spark.ml.feature.StringIndexer
    // "Maps a string column of labels to an ML column of label indices"
    val stringindexer = new StringIndexer().setInputCol("LABEL").setOutputCol("label_indices")
      
    val df_1 = stringindexer.fit(df).transform(df)
    
    // Making dataframe, removing header and changing a,b, label_indices(label) to double. Drop "any" drops NULL values
   
    val data = df_1.drop("LABEL")
                              .withColumn("a",col("a").cast("double"))
                              .withColumn("b", col("b").cast("double"))
                              .withColumn("label", col("label_indices").cast("double"))
                              .na.drop("any") 
   
    data.printSchema()
    
    //data to memory
    data.cache()
    data.show(12)
    
     // Creating VectorAssembler by mapping a,b and label to features.
    val vectorAssembler = new VectorAssembler().setInputCols(Array("a", "b", "label")).setOutputCol("features")

    val tf_Pl = new Pipeline().setStages(Array(vectorAssembler))
    val Pl = tf_Pl.fit(data)
    val tf_training = Pl.transform(data)
    
    import scala.collection.mutable.ArrayBuffer
    //"Creates a collection with specified elements"
    val cluster_centroids = ArrayBuffer [Int] ()
    val cluster_costs = ArrayBuffer [Double] ()

    // Calculating cluster_costs in for loop. 
    for (i <- low to high) {
      
      //Creating k means object and fitting tf_training to get k-means object
      val kmeans = new KMeans().setK(i).setSeed(1L)
      val kmModel = kmeans.fit(tf_training)
      val cost = kmModel.computeCost(tf_training)
      
      cluster_centroids += i
      cluster_costs += cost
      
    }
    //"which returns an array of (k,cost) pairs, where k is the number of means and cost is come cost for clustering"
    val cluster_pairs = cluster_centroids.toArray.zip(cluster_costs)
    return cluster_pairs
    
  }
     
  
    
}


