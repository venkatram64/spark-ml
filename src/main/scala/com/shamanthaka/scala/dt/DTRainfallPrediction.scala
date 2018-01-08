package com.shamanthaka.scala.dt

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession

/**
  * Created by Shamanthaka on 12/27/2017.
  */
object DTRainfallPrediction extends App{


  val sparkSession = SparkSession
    .builder()
    .master("local")
    .appName("RainfallPrediction")
    .getOrCreate()

  val testData = sparkSession.read.option("header", "true").option("inferSchema", "true").format("csv").load("weather-test.csv")
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

  val model = PipelineModel.load("dtWeatherModel")


  val predictions = model.transform(testData)

  predictions.printSchema()

  val predictionAndLabels = predictions.select($"prediction", $"label",$"probability")
  predictionAndLabels.show(100)

  sparkSession.stop()

}
