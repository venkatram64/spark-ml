package com.shamanthaka.scala

import java.io.{BufferedWriter, File, FileWriter}

import scala.io.Source

/**
  * Created by Shamanthaka on 1/27/2018.
  */
object CSVToLibSVM2 extends App{

  val file = new File("flowmeterd_libsvm_data")
  val bw = new BufferedWriter(new FileWriter(file))

  def concat(a: Array[String]):String ={
    var result = a(0) + " "
    for(i <- 1 to a.size.toInt - 1)
      result = result + i + ":" + a(i) + " "

    bw.write(result + "\n")
    return result
  }

  val lines = Source.fromFile("testd.txt").getLines()

  //val headerLIne = lines.take(1).next()

  for (l <- lines){
    //println("data without comma: " + l.split(",").toList)
    println(concat(l.split("\t")))
  }
  bw.close()

}
