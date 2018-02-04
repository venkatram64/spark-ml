package com.shamanthaka.scala.pca.dt

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Row, SparkSession}

/**
  * Created by Shamanthaka on 12/25/2017.
  */
object DTPCAHeartDiseaseModel extends App{

  val sparkSession = SparkSession
    .builder()
    .master("local")
    .appName("DTPCAHeartDiseaseModel")
    .getOrCreate()


  val data = sparkSession.read.format("libsvm").load("cleveland_heart_disease_libsvm.txt")
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

  // Index labels, adding metadata to the label column.
  // Fit on whole dataset to include all labels in index.
  val labelIndexer = new StringIndexer()
    .setInputCol("label")
    .setOutputCol("indexedLabel")
    .fit(data)

  val pca = new PCA()
    .setInputCol("features")
    .setOutputCol("pcaFeatures")
    .setK(10)     //10 principal components are chosen
    .fit(data)

  // Split the data into training and test sets (30% held out for testing).
  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

  // Train a DecisionTree model.
  val dt = new DecisionTreeClassifier()
    .setLabelCol("indexedLabel")
    .setFeaturesCol("pcaFeatures")

  // Convert indexed labels back to original labels.
  val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(labelIndexer.labels)

  // Chain indexers and forest in a Pipeline.
  val pipeline = new Pipeline()
    .setStages(Array(labelIndexer, pca, dt, labelConverter))

  // Train model. This also runs the indexers.
  val model = pipeline.fit(trainingData)

  model.write.overwrite().save("dtPCAHeatDiseaseModel");

  val predictions = model.transform(testData)

  predictions.printSchema()

  // Select example rows to display.
  //predictions.select("predictedLabel", "label", "probability", "features").show(300)

  predictions.select("prediction","label","probability", "pcaFeatures")
    .collect()
    .foreach{case Row(prediction: Double, label: Double, probability: Vector, pcaFeatures: Vector) =>
      println(s"($pcaFeatures, $label) -> prob = $probability, prediction=$prediction")
    }

  // Select (prediction, true label) and compute test error.
  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")

  val accuracy = evaluator.evaluate(predictions)
  println("Test Accuracy = " + accuracy * 100)
  println("Test Error = " + (1.0 - accuracy) * 100)

  val dtModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
  println("Learned classification forest model:\n" + dtModel.toDebugString)

  sparkSession.stop()

}
