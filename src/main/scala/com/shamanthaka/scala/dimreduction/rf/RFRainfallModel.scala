package com.shamanthaka.scala.dimreduction.rf

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

/**
  * Created by Shamanthaka on 12/25/2017.
  */
object RFRainfallModel extends App{

  val sparkSession = SparkSession
    .builder()
    .master("local")
    .appName("RFRainfallModel")
    .getOrCreate()


  val data = sparkSession.read.format("libsvm").load("weather_libsvm_data.txt")
  //show schema
  println("****** data schema will be printed ****. ")
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

  // Index labels, adding metadata to the label column.
  // Fit on whole dataset to include all labels in index.
  val labelIndexer = new StringIndexer()
    .setInputCol("label")
    .setOutputCol("indexedLabel")
    .fit(data)
  // Automatically identify categorical features, and index them.
  // Set maxCategories so features with > 10 distinct values are treated as continuous.
  val featureIndexer = new VectorIndexer()
    .setInputCol("features")
    .setOutputCol("indexedFeatures")
    .setMaxCategories(10)
    .fit(data)



  // Split the data into training and test sets (30% held out for testing).
  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

  // Train a RandomForest model.
  val rf = new RandomForestClassifier()
    .setLabelCol("indexedLabel")
    .setFeaturesCol("indexedFeatures")
    .setNumTrees(10)

  // Convert indexed labels back to original labels.
  val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(labelIndexer.labels)

  // Chain indexers and forest in a Pipeline.
  val pipeline = new Pipeline()
    .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

  // Train model. This also runs the indexers.
  val model = pipeline.fit(trainingData)

  model.write.overwrite().save("rfSampleModel2");

  val predictions = model.transform(testData)
  println("****** predicted data schema will be printed ****. ")
  predictions.printSchema()

  // Select example rows to display.
  predictions.select("prediction","label","probability", "features").show(100)

  // Select (prediction, true label) and compute test error.
  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
  val accuracy = evaluator.evaluate(predictions)
  println("Test Error = " + (1.0 - accuracy))

  val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
  println("Learned classification forest model:\n" + rfModel.toDebugString)

  sparkSession.stop()

}
