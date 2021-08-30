# Use getpass to obtain user netID
import getpass
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

    netid = getpass.getuser()
    
    user = ((train.select('user_id').distinct()).union(val.select('user_id').distinct())
            .union(test.select('user_id').distinct())).distinct()
    track = ((train.select('track_id').distinct()).union(val.select('track_id').distinct())
            .union(test.select('track_id').distinct())).distinct()
            
    user.write.format('parquet').mode('overwrite').save('hdfs:/user/{}/df_user2.parquet'.format(netid))
    track.write.format('parquet').mode('overwrite').save('hdfs:/user/{}/df_track2.parquet'.format(netid))

    train.write.format('parquet').mode('overwrite').save('hdfs:/user/{}/df_train2.parquet'.format(netid))
    val.write.format('parquet').mode('overwrite').save('hdfs:/user/{}/df_val2.parquet'.format(netid))
    test.write.format('parquet').mode('overwrite').save('hdfs:/user/{}/df_test2.parquet'.format(netid))

    #Subsample======================================================
    val1 = val.withColumnRenamed('user_id','user_id_val').select('user_id_val').distinct()
    q1 = train.join(val1,train.user_id == val1.user_id_val,how = 'inner').select('user_id','count','track_id')
    q2 = train.select('user_id','count','track_id').subtract(q1)
    
    #1% subsample
    subsample1 = q2.sample(withReplacement=False, fraction=0.01, seed = 233).union(q1)
    subsample1.show()
    print(subsample1.count())
    subsample1.write.format('parquet').mode('overwrite').save('hdfs:/user/{}/subsample12.parquet'.format(netid))
    
    #5% subsample
    subsample5 = q2.sample(withReplacement=False, fraction=0.05, seed = 233).union(q1)
    subsample5.show()
    print(subsample5.count())
    subsample5.write.format('parquet').mode('overwrite').save('hdfs:/user/{}/subsample52.parquet'.format(netid))

    #25% subsample
    subsample25 = q2.sample(withReplacement=False, fraction=0.25, seed = 233).union(q1)
    subsample25.show()
    print(subsample25.count())
    subsample25.write.format('parquet').mode('overwrite').save('hdfs:/user/{}/subsample252.parquet'.format(netid))


if __name__ == "__main__":
    # Create the spark session object
    MAX_MEMORY = "20g"

    spark = SparkSession \
    .builder \
    .appName("ttt") \
    .config("spark.executor.memory", MAX_MEMORY) \
    .config("spark.driver.memory", MAX_MEMORY) \
    .getOrCreate()
    # Call our main routine
    main(spark)