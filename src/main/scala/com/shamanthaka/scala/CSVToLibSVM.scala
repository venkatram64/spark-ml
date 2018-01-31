package com.shamanthaka.scala

import org.apache.spark.sql.SparkSession

import scala.io.Source

/**
  * Created by Shamanthaka on 1/27/2018.
  */
object CSVToLibSVM extends App{


  def concat(a: Array[String]):String ={
    var result = a(0) + " "
    for(i <- 1 to a.size.toInt - 1)
      result = result + i + ":" + a(i) + " "
    return result
  }

  val lines = Source.fromFile("cleveland-heart-disease.csv").getLines()

  val headerLIne = lines.take(1).next()

  for (l <- lines){
    //println("data without comma: " + l.split(",").toList)
    println(concat(l.split(",")))
  }


}
