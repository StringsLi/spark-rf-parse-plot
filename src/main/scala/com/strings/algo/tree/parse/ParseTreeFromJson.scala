package com.strings.algo.tree.parse

import java.io.File
import scala.collection.mutable.ArrayBuffer
import com.strings.algo.tree.util.GsonUtil
import scala.io.Source

case class decisionNode(featureIndex:Option[Int],
                        gain:Option[Double],
                        impurity:Double,
                        threshold:Option[Double], // Continuous split
                        nodeType:String, // Internal or leaf
                        impurityStats:Array[Double],
                        splitType: Option[String], // Continuous and categorical
                        leftCategories: Option[Array[Double]], // Categorical Split
                        rightCategories: Option[Array[Double]], // Categorical Split
                        prediction:Double,
                        leftChild:Option[decisionNode],
                        rightChild:Option[decisionNode])

class ParseTreeFromJson{
    var treeNodeList:ArrayBuffer[decisionNode] = new ArrayBuffer[decisionNode]() //一定要初始化，构造的决策树列表
    var nums_class:Int = 2  //分几类问题

    def load(path:String): Unit = {
      val filesIter = subdirs(path)
      while(filesIter.hasNext){
        val file = filesIter.next()
        val data = GsonUtil.toMapString(Source.fromFile(file).mkString)
        val node = walk(data)
        treeNodeList.append(node)
      }
    }

  def subdirs(path:String):Iterator[File] = {
    val dir: File  = new File(path)
    dir.listFiles.filter(_.isFile).toIterator
  }

  def string2arr(str:String):Array[Double] = {
     str.replace("[","")
        .replace("]","")
        .split(",")
        .map(_.toDouble)
  }

  def walk(data:java.util.Map[String,String]): decisionNode = {
    val node_type = data.get("nodeType")
    val gain:Option[Double] = node_type match {
      case "internal" => Some(data.get("gain").toDouble)
      case "leaf" => None
    }
    val feature_index:Option[Int] = node_type match {
      case "internal" => Some(data.get("featureIndex").toDouble.toInt)
      case "leaf" => None
    }

    val split_type:Option[String] = node_type match {
      case "internal" => Some(data.get("splitType"))
      case "leaf" => None
    }

    val node_threshold:Option[Double] = split_type match {
      case Some("continuous") => Some(data.get("threshold").toDouble)
      case Some("categorical") => None
      case _ => None
    }

    val (left_categories:Option[Array[Double]],
    right_categories:Option[Array[Double]]) = split_type match {
      case Some("categorical") => (Some(string2arr(data.get("leftCategories"))),
        Some(string2arr(data.get("rightCategories")))
      )
      case _ => (None,None)
    }

    val left_child: Option[decisionNode] = node_type match {
      case "internal" => Some(walk(GsonUtil.toMapString(data.get("leftChild"))))
      case _ => None
    }

    val right_child: Option[decisionNode] = node_type match {
      case "internal" => Some(walk(GsonUtil.toMapString(data.get("rightChild"))))
      case _ => None
    }
    decisionNode(featureIndex = feature_index,
      gain = gain,
      impurity = data.get("impurity").toDouble,
      threshold = node_threshold,
      nodeType = node_type,
      impurityStats = string2arr(data.get("impurityStats")),
      splitType = split_type,
      leftCategories = left_categories, // Categorical Split
      rightCategories = right_categories, // Categorical Split
      prediction = data.get("prediction").toDouble,
      leftChild = left_child,
      rightChild = right_child
    )
  }

  /**
   * 预测概率
   * @param feature_map
   * @return
   */
  def predict_pro_one(feature_map:Map[Int,Double]): Array[Double] ={
    val raw_predict = predict_raw(feature_map)
    val predict_sum = raw_predict.sum
    raw_predict.map(_/predict_sum)
  }

  /**
   *
   * @param feature_map
   * @return
   */
  def predict_raw(feature_map:Map[Int,Double]): Array[Double] ={
    val raw_ret = Array.fill(nums_class)(0.0)
    for(tree <- treeNodeList){
      var curTree = tree.copy()
      while(curTree.nodeType.equals("internal")){
        if(feature_map.contains(curTree.featureIndex.get)){
          if(curTree.splitType.get.equals("continuous")){
            if(feature_map(curTree.featureIndex.get) <= curTree.threshold.get){
              curTree = curTree.leftChild.get
            }else{
              curTree = curTree.rightChild.get
            }
          }else{
              if(curTree.leftCategories.get.contains(feature_map(curTree.featureIndex.get))){
                curTree = curTree.leftChild.get
              }else{
                curTree = curTree.rightChild.get
              }
          }
        }
      }
      val impurity_stats = curTree.impurityStats
      val total = impurity_stats.sum
      var i = 0
      while(i < nums_class) {
        raw_ret(i) += impurity_stats(i) / total
        i += 1
      }
    }
    raw_ret
  }

  /**
   *  读取数据
   * @param file_name 文件名称，单行json文件
   * @param columns2index 数据列名和数据列index之间的对应关系
   * @return
   */
  def read_data(file_name:String,columns2index:Map[String,Int]):List[Map[Int,Double]]= {
    val res:ArrayBuffer[Map[Int,Double]] = new ArrayBuffer[Map[Int, Double]]()

    Source.fromFile(file_name).getLines().toList.foreach{line =>
     val dict_data =  GsonUtil.toMapString(line)
      var res_map:Map[Int,Double] = Map()
      dict_data.keySet().toArray().foreach{key =>
        val keyString = key.toString
        res_map += (columns2index.get(keyString).get -> dict_data.get(keyString).toDouble)
      }
      res.append(res_map)
    }

    res.toList
  }

}
