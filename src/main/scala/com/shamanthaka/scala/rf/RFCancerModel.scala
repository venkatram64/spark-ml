package com.shamanthaka.scala.rf

import com.shamanthaka.scala.dimreduction.rf.RFRainfallModel.{evaluator, model, predictions}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

/**
  * Created by Shamanthaka on 12/25/2017.
  *
  * 1. Sample code number: id number
    2. Clump Thickness: 1 - 10
    3. Uniformity of Cell Size: 1 - 10
    4. Uniformity of Cell Shape: 1 - 10
    5. Marginal Adhesion: 1 - 10
    6. Single Epithelial Cell Size: 1 - 10
    7. Bare Nuclei: 1 - 10
    8. Bland Chromatin: 1 - 10
    9. Normal Nucleoli: 1 - 10
    10. Mitoses: 1 - 10
    11. Class: (2 for benign, 4 for malignant)
  */
//https://mapr.com/blog/predicting-breast-cancer-using-apache-spark-machine-learning-logistic-regression/
object RFCancerModel extends App{

  val sparkSession = SparkSession
    .builder()
    .master("local")
    .appName("RFCancerModel")
    .getOrCreate()

  val data = sparkSession.read.option("header", "true").option("inferSchema", "true").format("csv").load("cancer.csv")
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

  val logregDataAll = data.select($"clas",
    $"clump_thick_ness",$"cell_size", $"cell_shape", $"adhesion", $"epitherlial_cell_size",$"bare_nuclei",
    $"bland_chromatin",$"normal_nucleoli",$"clas"
  )

  val logregData = logregDataAll.na.drop()


  val rainTomorrowIndexer = new StringIndexer().setInputCol("clas").setOutputCol("label")
  val rainTomorrowEncoder = new OneHotEncoder().setInputCol("label").setOutputCol("CancerVec")


  val assembler = new VectorAssembler().
    setInputCols(Array("clump_thick_ness","cell_size","adhesion","epitherlial_cell_size","bare_nuclei","bland_chromatin",
      "normal_nucleoli","normal_nucleoli","clas"))
    .setOutputCol("features")



  val Array(training, test) = logregData.randomSplit(Array(0.7, 0.3), seed=12345)

  println ("training data count " + training.count)
  println ("test data count " + test.count)

  val rf = new RandomForestClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setImpurity("gini")
    .setMaxDepth(3)
    .setNumTrees(10)

  val pipeline = new Pipeline().setStages(Array(rainTomorrowIndexer, rainTomorrowEncoder,assembler,rf))


  val model = pipeline.fit(training)

  model.write.overwrite().save("rfCancerModel");

  val predictions = model.transform(test)

  predictions.printSchema()

  val predictionAndLabels = predictions.select($"prediction", $"label",$"probability")
  predictionAndLabels.show(100)

  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")

  val accuracy = evaluator.evaluate(predictions)
  println("Test Error = " + (1.0 - accuracy))

  val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
  println("Learned classification forest model:\n" + rfModel.toDebugString)

  sparkSession.stop()

}
