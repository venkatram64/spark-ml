package com.shamanthaka.scala.lr

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

/**
  * Created by Shamanthaka on 12/25/2017.
  */
object LRRainfallModel extends App{


  val sparkSession = SparkSession
    .builder()
    .master("local")
    .appName("LRRainfallModel")
    .getOrCreate()

  val data = sparkSession.read.option("header", "true").option("inferSchema", "true").format("csv").load("weather.csv")
  //show schema
  data.printSchema()

  val colnames = data.columns
  val firstrow = data.head(1)(0)
  println("\n")
  println("Example Data Row")

  for(ind <- Range(1, colnames.length)){
    println(colnames(ind))
    println(firstrow(ind))
    println("\n")
  }

  import sparkSession.implicits._

  val logregDataAll = data.select($"RainTomorrow",
    $"MinTemp",$"MaxTemp", $"Rainfall", $"Evaporation", $"Sunshine",$"WindGustSpeed",
    $"WindDir9am",$"WindDir3pm",$"WindSpeed9am", $"WindSpeed3pm",$"Humidity9am",$"Humidity3pm",
    $"Pressure9am", $"Pressure3pm", $"Cloud9am",$"Cloud3pm",$"Temp9am", $"Temp3pm",  $"RISK_MM"
  )

  val logregData = logregDataAll.na.drop()


  val rainTomorrowIndexer = new StringIndexer().setInputCol("RainTomorrow").setOutputCol("label")
  val rainTomorrowEncoder = new OneHotEncoder().setInputCol("label").setOutputCol("RainTomorrowVec")


  val assembler = new VectorAssembler().
    setInputCols(Array("MinTemp","MaxTemp","Rainfall","Evaporation","Sunshine","WindGustSpeed",
      "WindSpeed9am","WindSpeed3pm","Humidity9am","Humidity3pm","Pressure9am",
      "Pressure3pm","Cloud9am","Cloud3pm","Temp9am","Temp3pm","RISK_MM"))
    .setOutputCol("features")


  val Array(training, test) = logregData.randomSplit(Array(0.7, 0.3), seed=12345)

  println ("training data count " + training.count)
  println ("test data count " + test.count)


  val lr = new LinearRegression()
    .setMaxIter(10)
    .setRegParam(0.001)
    .setLabelCol("label")
    .setFeaturesCol("features")

  val pipeline = new Pipeline().setStages(Array(rainTomorrowIndexer, rainTomorrowEncoder,assembler,lr))


  val model = pipeline.fit(training)

  model.write.overwrite().save("lrWeatherModel");

  val predictions = model.transform(test)

  predictions.printSchema()

  val predictionAndLabels = predictions.select($"prediction", $"label")
  predictionAndLabels.show(100)


  sparkSession.stop()

}
