package com.shamanthaka.scala.pca.rf

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.{SparkSession,Row}
import org.apache.spark.ml.linalg.{Vector, Vectors}

/**
  * Created by Shamanthaka on 12/27/2017.
  */
object RFPCARainfallPrediction extends App{


  val sparkSession = SparkSession
    .builder()
    .master("local")
    .appName("RFPCARainfallPrediction")
    .getOrCreate()

  val testData = sparkSession.read.format("libsvm").load("weather_libsvm_test_data.txt")
  //show schema
  println("****** data schema will be printed ****. ")
  testData.printSchema()

  val colnames = testData.columns
  val firstrow = testData.head(1)(0)
  println("\n")
  println("Example Data Row")

  for(ind <- Range(1, colnames.length)){
    println(colnames(ind))
    println(firstrow(ind))
    println("\n")
  }


  import sparkSession.implicits._

  val model = PipelineModel.load("rfPCARAINFALLModel")



  val predictions = model.transform(testData)
  println("****** predicted data schema will be printed ****. ")
  predictions.printSchema()

/*  val predictionAndLabels = predictions.select($"prediction", $"label",$"probability")
  predictionAndLabels.show(100)*/

  //predictions.select($"prediction", $"label",$"probability",$"pcaFeatures").show(300)

  predictions.select("prediction","label","probability", "pcaFeatures")
    .collect()
    .foreach{case Row(prediction: Double, label: Double, probability: Vector, pcaFeatures: Vector) =>
      println(s"($pcaFeatures, $label) -> prob = $probability, prediction=$prediction")
    }

  sparkSession.stop()

}
