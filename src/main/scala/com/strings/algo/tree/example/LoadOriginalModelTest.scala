package com.strings.algo.tree.example

/**
 * load spark 保存的原始模型，来对比
 *
 */

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.SparkSession

object LoadOriginalModelTest {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local")
      .appName("LoadOriginalModelTest")
      .config("spark.sql.warehouse.dir", "file:///E:/spark-warehouse")
      .getOrCreate()

    val resource_path = "C:\\Users\\morni\\Desktop\\github\\spark-random-forest-parse\\src\\main\\resources"

    val data = spark.read.json(resource_path + "/data/iris.json")

    val cols = data.columns.map(f => col(f).cast(DoubleType))

    val data2 = data.select(cols:_*)

    val vectorAssembler = new VectorAssembler()
      .setInputCols(data.columns.tail)
      .setOutputCol("features")

    val assemblerData = vectorAssembler.transform(data2)

    val model = PipelineModel.load(resource_path + "\\orignal_model")

    val predictions = model.transform(assemblerData)

    predictions.show(100)
    // Select example rows to display.
    predictions.select("prediction", "class", "features","probability").show(100,truncate = false)


    val treeModel = model.stages(1).asInstanceOf[RandomForestClassificationModel]
    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
  }
}


