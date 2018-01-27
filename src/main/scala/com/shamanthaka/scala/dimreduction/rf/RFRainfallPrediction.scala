package com.shamanthaka.scala.dimreduction.rf

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{LabeledPoint, OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions._

/**
  * Created by Shamanthaka on 12/27/2017.
  */
object RFRainfallPrediction extends App{


  val sparkSession = SparkSession
    .builder()
    .master("local")
    .appName("RFRainfallPrediction")
    .getOrCreate()

  val testData = sparkSession.read.format("libsvm").load("weather_libsvm_test_data.txt")
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

  val model = PipelineModel.load("rfSampleModel2")



  val predictions = model.transform(testData)

  predictions.printSchema()

/*  val predictionAndLabels = predictions.select($"prediction", $"label",$"probability")
  predictionAndLabels.show(100)*/

  predictions.select($"prediction", $"label",$"probability").show(100)
  sparkSession.stop()

}
