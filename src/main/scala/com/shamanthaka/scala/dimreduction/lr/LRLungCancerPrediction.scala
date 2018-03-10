package com.shamanthaka.scala.dimreduction.lr

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession

/**
  * Created by Shamanthaka on 12/27/2017.
  */
object LRLungCancerPrediction extends App{

  val sparkSession = SparkSession
    .builder()
    .master("local")
    .appName("LRLungCancerPrediction")
    .getOrCreate()

  val testData = sparkSession.read.format("libsvm").load("lungcancer_libsvm_data.txt")
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

  val model = PipelineModel.load("lrLungCancerModel")


  val predictions = model.transform(testData)

  predictions.printSchema()

  val predictionAndLabels = predictions.select($"prediction", $"label",$"probability")
  predictionAndLabels.show(100)

  sparkSession.stop()

}