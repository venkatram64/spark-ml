package com.shamanthaka.scala.pca.lr

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Row, SparkSession}

/**
  * Created by Shamanthaka on 12/27/2017.
  */
object LRPCAHeartDiseasePrediction extends App{

  val sparkSession = SparkSession
    .builder()
    .master("local")
    .appName("LRPCAHeartDiseasePrediction")
    .getOrCreate()

  val testData = sparkSession.read.format("libsvm").load("cleveland_heart_disease_libsvm_test.txt")
  //show schema
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

  val model = PipelineModel.load("lrPCAHeatDiseaseModel")


  val predictions = model.transform(testData)

  predictions.printSchema()

  /*val predictionAndLabels = predictions.select($"prediction", $"label",$"probability")
  predictionAndLabels.show(100)*/
  predictions.select("prediction","label","probability", "pcaFeatures")
    .collect()
    .foreach{case Row(prediction: Double, label: Double, probability: Vector, pcaFeatures: Vector) =>
      println(s"($pcaFeatures, $label) -> prob = $probability, prediction=$prediction")
    }

  sparkSession.stop()

}
