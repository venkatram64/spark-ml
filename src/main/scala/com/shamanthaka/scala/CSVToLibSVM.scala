package com.shamanthaka.scala

import org.apache.spark.sql.SparkSession

import scala.io.Source

/**
  * Created by Venkatram on 1/27/2018.
  */
object CSVToLibSVM extends App{

  val sparkSession = SparkSession
    .builder()
    .master("local")
    .appName("RFRainfallModel")
    .getOrCreate()

  def concat(a:Array[String]):String ={
    var result=a(0)+" "
    for(i<-1 to a.size.toInt-1)
      result=result+i+":"+a(i)(0)+" "
    return result
  }

  val src = Source.fromFile("weather2.csv").getLines()

  val headerLIne = src.take(1).next()

  for (l <- src){
    println(concat(l.split(",")))
  }

  /*val rfile = sparkSession.read.option("header", "false").option("inferSchema", "true").format("csv").load("weather2.csv")
  val f=rfile.map(line => line.toString().split(",")).map(i=>concat(i))*/

}
