package com.shamanthaka.scala.pca.rf

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.linalg.{Vector, Vectors}

/**
  * Created by Shamanthaka on 12/27/2017.
  */
object RFPCACancerPrediction extends App{

  val sparkSession = SparkSession
    .builder()
    .master("local")
    .appName("RFPCACancerPrediction")
    .getOrCreate()

  val testData = sparkSession.read.format("libsvm").load("cancer_libsvm_test_data.txt")
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

  val model = PipelineModel.load("rfPCACancerModel")

  val predictions = model.transform(testData)
  println("****** predicted data schema will be printed ****. ")
  predictions.printSchema()

/*  val predictionAndLabels = predictions.select($"prediction", $"label",$"probability")
  predictionAndLabels.show(100)*/

  //predictions.select($"prediction", $"label",$"probability").show(300)

  predictions.select("prediction","label","probability", "pcaFeatures")
    .collect()
    .foreach{case Row(prediction: Double, label: Double, probability: Vector, pcaFeatures: Vector) =>
      println(s"($pcaFeatures, $label) -> prob = $probability, prediction=$prediction")
    }

  sparkSession.stop()

}
