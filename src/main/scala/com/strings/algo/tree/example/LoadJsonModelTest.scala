package com.strings.algo.tree.example

import com.strings.algo.tree.parse.ParseTreeFromJson

/**
 * 测试基于json文件构建的随机森林模型预测的准确性
 */

object LoadJsonModelTest {

  def main(args: Array[String]): Unit = {

    val resource_path = "C:\\Users\\morni\\Desktop\\github\\spark-random-forest-parse\\src\\main\\resources"

    // 加载json 模型文件
    val parse = new ParseTreeFromJson()
    parse.load(resource_path + "\\json_model\\")

    val  testfilename = resource_path + "\\data\\iris.json"
    var cols2index:Map[String,Int] = Map()

    val cols = Array("petal length (cm)",
      "petal width (cm)","sepal length (cm)","sepal width (cm)","class")

    cols.zipWithIndex.foreach {x =>
      val value = x._1
      val index = x._2
      cols2index += (value -> index)
    }
    val test_data = parse.read_data(testfilename,cols2index)

    for(dict <- test_data) {
      val pred2 = parse.predict_pro_one(dict)
      println(dict,pred2.toList)
    }
  }


}
