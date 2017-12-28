package com.shamanthaka.scala;

import org.junit.Test
import org.junit.runner.RunWith
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.test.context.ActiveProfiles
import org.springframework.test.context.junit4.SpringRunner;

@ActiveProfiles(Array("test"))
@RunWith(classOf[SpringRunner])
//@SpringBootTest(classes = Array(classOf[PhoenixConnection], classOf[EmployeeService]))
@EnableConfigurationProperties
class ScalaApplicationTests {

	@Test
	def myTest(): Unit = {
		println("Hi...")
	}

}
