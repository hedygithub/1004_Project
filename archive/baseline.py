# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession


def main(spark):
    '''Main routine for Project
    Parameters
    ----------
    spark : SparkSession object
    '''
    print('Begin loading data')

    # Load the Data
    train = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/cf_train.parquet')
    train.createOrReplaceTempView('train')
    val = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/cf_validation.parquet')
    val.createOrReplaceTempView('val')
    test = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/cf_test.parquet')
    test.createOrReplaceTempView('test')
    print('Successfully loaded the data')

    #
    train_counts = train.count()
    val_counts = val.count()
    test_counts = test.count()
    print('train counts: {}, val counts: {}, test counts: {}'.format(train_counts, val_counts, test_counts))


    #Encode string to index============================================
    from pyspark.ml.feature import StringIndexer
    from pyspark.sql.types import IntegerType

    user = (train.select('user_id').union(val.select('user_id'))
            .union(test.select('user_id'))).distinct()
    track = (train.select('track_id').union(val.select('track_id'))
            .union(test.select('track_id'))).distinct()


    indexer1 = StringIndexer(inputCol="user_id", outputCol="user_index")
    tran_user = indexer1.fit(user)
    indexer2 = StringIndexer(inputCol="track_id", outputCol="track_index")
    tran_track = indexer2.fit(track)

    train = tran_user.transform(train)
    val = tran_user.transform(val)
    test = tran_user.transform(test)

    train = tran_track.transform(train)
    val = tran_track.transform(val)
    test = tran_track.transform(test)
    
    train = train.withColumn("user_index", train["user_index"].cast(IntegerType()))\
            .withColumn("track_index", train["track_index"].cast(IntegerType()))
    val = val.withColumn("user_index", val["user_index"].cast(IntegerType()))\
            .withColumn("track_index", val["track_index"].cast(IntegerType()))
    test = test.withColumn("user_index", test["user_index"].cast(IntegerType()))\
            .withColumn("track_index", test["track_index"].cast(IntegerType()))
    
    #Subsample======================================================
    val1 = val.withColumnRenamed('user_id','user_id_val').select('user_id_val').distinct()
    q1 = train.join(val1,train.user_id == val1.user_id_val,how = 'inner').select('user_index','count','track_index')
    q_sub = train.join(val1,train.user_id == val1.user_id_val,how = 'left_anti').select('user_index','count','track_index')    
     
    #1% subsample
    q2=q_sub.sample(withReplacement=False, fraction=0.01)
    subsample1 = q2.union(q1)
    subsample1.show()
    print(subsample1.count())
    
    #5% subsample
    q3=q_sub.sample(withReplacement=False, fraction=0.05)
    subsample5 = q3.union(q1)
    subsample5.show()
    print(subsample5.count())

    #25% subsample
    q4=q_sub.sample(withReplacement=False, fraction=0.25)
    subsample25 = q4.union(q1)
    subsample25.show()
    print(subsample25.count())

    #=============================================================
#     from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
#     from pyspark.ml.evaluation import RegressionEvaluator
#     from pyspark.ml.recommendation import ALS
# 
#     training = subsample1
#     testing = val
# 
#     als = ALS(maxIter=5,numBlocks=1000, rank = 10, regParam=0.01, userCol="user_index", itemCol="track_index", ratingCol="count",
#               coldStartStrategy="drop",implicitPrefs = True)
#     model = als.fit(training)
# 
#     # Evaluate the model by computing the RMSE on the test data
#     predictions = model.transform(testing)
#     evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
#                                     predictionCol="prediction")
#     rmse = evaluator.evaluate(predictions)
#     print("Root-mean-square error = " + str(rmse))

if __name__ == "__main__":
    # Create the spark session object
    MAX_MEMORY = "20g"

    spark = SparkSession \
    .builder \
    .appName("Foo") \
    .config("spark.executor.memory", MAX_MEMORY) \
    .config("spark.driver.memory", MAX_MEMORY) \
    .getOrCreate()
    # Call our main routine
    main(spark)
