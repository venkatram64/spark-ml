package com.shamanthaka.scala.nd.rf

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession

/**
  * Created by Shamanthaka on 12/27/2017.
  */
object RFYearPrediction extends App{


  val sparkSession = SparkSession
    .builder()
    .master("local")
    .appName("RFYearPrediction")
    .getOrCreate()

  val testData = sparkSession.read.format("libsvm").load("year1_libsvm_test_data.txt")
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

  val model = PipelineModel.load("rfYearModel")



  val predictions = model.transform(testData)
  println("****** predicted data schema will be printed ****. ")
  predictions.printSchema()

/*  val predictionAndLabels = predictions.select($"prediction", $"label",$"probability")
  predictionAndLabels.show(100)*/

  predictions.select($"prediction", $"label",$"probability").show(300)

/*  predictions.select("prediction","label","probability", "features")
    .collect()
    .foreach{case Row(prediction: Double, label: Double, probability: Vector, features: Vector) =>
      println(s"($features, $label) -> prob = $probability, prediction=$prediction")
    }*/

  sparkSession.stop()

}
