package com.strings.algo.tree.save

import java.io.{File, PrintWriter}

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.tree.SparkMLTree
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType

object RandomForestSaveModelExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("RandomForestSaveModelExample")
      .config("spark.sql.warehouse.dir", "file:///E:/spark-warehouse")
      .getOrCreate()

    val resource_path = "C:\\Users\\morni\\Desktop\\github\\spark-random-forest-parse\\src\\main\\resources"

    val data = spark.read.json(resource_path + "/data/iris.json")
    data.printSchema()

    val cols = data.columns.map(f => col(f).cast(DoubleType))

    val data2 = data.select(cols:_*)

    val vectorAssembler = new VectorAssembler()
      .setInputCols(data.columns.tail)
      .setOutputCol("features")

    val assemblerData = vectorAssembler.transform(data2)

    assemblerData.show(100)

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = assemblerData.randomSplit(Array(0.9, 0.1))

    // Train a DecisionTree model.
    val dt = new RandomForestClassifier()
      .setLabelCol("class")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(15)
      .setSeed(30)

    // Chain indexers and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, dt))

    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)

    val originalModelPath = resource_path + "/orignal_model/"
    model.save(originalModelPath)

    // Make predictions.
    val predictions = model.transform(testData)

    predictions.show(30)
    // Select example rows to display.
    predictions.select("prediction", "class", "features","probability").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("class")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${1.0 - accuracy}")

    var count = 0
    val filePath = resource_path + "/json_model/"
    model.stages(1).asInstanceOf[RandomForestClassificationModel].trees.foreach { x =>
      val sparkMLTree = new SparkMLTree(x)
      val ss = sparkMLTree.toJsonPlotFormat()
      write2json(ss,filePath,"tree_"+count + ".json")
      count += 1
    }

    def write2json(string:String,filePath:String,fileName:String) = {
      val writer  = new PrintWriter(new File(filePath+ fileName))
      for (line <- string){
        writer.write(line)
      }
      writer.close()
    }


    val treeModel = model.stages(1).asInstanceOf[RandomForestClassificationModel]
    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

    spark.stop()
  }
}
