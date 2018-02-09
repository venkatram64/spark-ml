package com.shamanthaka.scala

/**
  * Created by Venkatram on 2/8/2018.
  */
class OptionTest {
  //some or none is the option
  def listToLower(list: List[String]) = list.map(Option(_)).flatMap(x => x).map(_.toLowerCase)

  def listToLower2(list: List[String]) = list.map(_.toLowerCase)

  def listToLower3(list: List[String]) =
    list.map(item => if(item != null){
                  item.toLowerCase})


}

object OptionTest extends App{
  val hw = Option("Hello World")
  println(hw)
  val nil = Option(null)
  println(nil)

  val list = List("mY", "NamE","iS", "VeNkAtRaM",null)

  val optionTest = new OptionTest()
  //val toLower = optionTest.listToLower2(list)
  //val toLower = optionTest.listToLower3(list)
  val toLower = optionTest.listToLower(list)

  toLower.map(println)


}
