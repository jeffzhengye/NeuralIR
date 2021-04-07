1. pair 对如何获取
2. how to stop training?  - a) loss, b) evaluation metrics
3. 数据预处理方法确认 - 
4. histogram 生产确认：方法
5. output: tanh 
6. oov的处理: cosine=1, 
7. word2vec的处理：stop stemed
TODO list: 
    1) split training data, to get val data; or find best para on test for prove of idea




代码分为两部分：预处理阶段和网络训练阶段

网络训练阶段：

进入neural-ranking文件夹  

sh run_5folds_juno.sh 

cd result

cat *result >> merge

python prepare_trec_eval.py /result/merge ../data/robust04.bm25.noqe.res

预处理阶段：

进入 preprocessing文件夹

python transform_raw_topic_to_title_only.py  数据集的查询 doStem（可选）

python transform_raw_trec_to_single_file.py  数据集的文档 doStem（可选）

python run_word2vec - test.py  ../data/corpus

python generate_pair_histograms.py ../data/corpus/trec_corpus.stemmed.txt ../data/corpus/query.stemmed ../data/robust04.bm25.noqe.res ../data/trec_corpus.model 30 prerank

python generate_pair_histograms.py ../data/corpus/trec_corpus.stemmed.txt ../data/corpus/query.stemmed ../data/qrels.rob04.txt ../data/trec_corpus.model 30 qrel



data文件夹：存放预处理步骤的数据

5-folds文件夹 ： 训练和测试文档分类

prerank_histogram_30.txt ： 预排序文档的频率直方图

qrel_histogram_30.txt ： 相关/不相关文档频率直方图

qrels.rob04.txt ：相关/不相关描述文档

robust04.bm25.noqe.res ： 预排序文档

query.stemmed ： 查询文档

trec_corpus.stemmed.txt ： 文件文档



neural-ranking文件夹：存放DRMM的网络模型

models文件夹：存放DRMM生成的模型

result文件夹：存放DRMM生成的结果

keras_model.py ： DRMM网络代码

load_data.py ： 载入数据代码

loss_function.py ： 损失函数代码

prepare_trec_eval.py ：在进行evaluation前处理结果文档

run_5folds_juno.sh ： 运行5个分开的训练和测试得到五个结果

run_model.py ： 运行模型文件



preprocessing文件夹 ： 存放预处理步骤代码

check_oov_words.py ： 检查oov词汇

create_train_test_folds.py ： 生成5次拆分和1-0对用于训练和5次拆分用于测试

generate_pair_histograms.py ： 生成频率直方图

run_word2vec - test.py ： 获得词向量

TextCollection.py ：生成idf

transform_raw_topic_to_title_only.py ： 处理查询

transform_raw_trec_to_single_file.py ： 处理文档
