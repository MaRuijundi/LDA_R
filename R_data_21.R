library(tm)
library(SnowballC)
library(topicmodels)
#设置工作目录（根据需要修改路径）
setwd("C:\\Myproject_LDA\\database4\\data21_2486") 
#加载文档到语料库
#获取目录中的.txt文件列表
filenames <- list.files(getwd(),pattern="*.txt")
#将文件读入字符向量
files <- lapply(filenames,readLines,encoding="UTF-8")
#创建矢量语料库
articles.corpus <- VCorpus(VectorSource(files))
# 将每个字母变成小写
articles.corpus <- tm_map(articles.corpus, content_transformer ( tolower ))
# 删除标点符号
articles.corpus <- tm_map(articles.corpus, removePunctuation)
#删除数字
articles.corpus <- tm_map(articles.corpus, removeNumbers);
#删除空白格
articles.corpus <- tm_map(articles.corpus, stripWhitespace);
# 删除通用和自定义的停用词
stopword <- c(stopwords('english'),"fn","pt","au","af","sn","ei","pd","py","vl","su","bp","ep","di","ut","er", "ti","so","ab","can","one","and","like","just","gonna","know","really","right","going","get","well","lot","actually","new",
              "will","much","way","and","see","make","look",
              "also","able","say","back","got","take","great",
              "many","next","using","around","thing","two",
              "looking","said","kind","come","put","yeah",
              "even","still","ago","every","three","five","gonna",
              "okay","whether","seen","you","six","there","this",
              "and","but","you","want","thats","but","you",
              "folks","sure","run","and");
articles.corpus <- tm_map(articles.corpus, removeWords, stopword)
articles.corpus <- tm_map(articles.corpus, stemDocument)

#Create document-term matrix
params <- list(minDocFreq = 1,removeNumbers = TRUE,stopwords = TRUE,stemming = TRUE,weighting = weightTf)
articleDtm <- DocumentTermMatrix(articles.corpus, control = params);
# Convert rownames to filenames
rownames(articleDtm) <- filenames
# Collapse matrix by summing over columns
freq <- colSums(as.matrix(articleDtm))
#Length should be total number of terms
length(freq)
# Create sort order (descending)
ord <- order(freq,decreasing=TRUE)
# List all terms in decreasing order of freq and write to disk
#inspect most frequently occurring terms
freq[head(ord)]
#inspect least frequently occurring terms
freq[tail(ord)]
#total terms
freq[ord]
#list most frequent terms. Lower bound specified as second argument
findFreqTerms(articleDtm,lowfreq = 600)
write.csv(freq[ord],"word_freq.csv")


#histogram 直方图
wf= data.frame(term=names(freq),occurrences=freq)
library(ggplot2)
p <- ggplot(subset(wf,freq>1000),aes(term,occurrences))
p <- p + geom_bar(stat="identity")
p <- p + theme(axis.text.x = element_text(angle = 45,hjust = 1))
p

#wordcloud 云图
library(wordcloud)
set.seed(42)
#limit words by specifying min frequency
wordcloud(names(freq),freq,min.freq = 850)
wordcloud(names(freq),freq,min.freq = 850,color = brewer.pal(6,"Dark2"))

# Load topic models library
library(topicmodels)

# Set parameters for Gibbs sampling
burnin <- 4000
iter <- 2000
thin <- 500
seed <-list(2003,5,63,100001,765)
nstart <- 5
best <- TRUE

# Number of topics
k <- 6

# Run LDA using Gibbs sampling
ldaOut <-LDA(articleDtm,k, method="Gibbs", control=list(nstart=nstart, seed = seed, best=best, burnin = burnin, iter = iter, thin=thin))

# write out results
# docs to topics

ldaOut.topics <- as.matrix(topics(ldaOut))
write.csv(ldaOut.topics,file=paste("LDAGibbs",k,"DocsToTopics.csv"))

# Top N terms in each topic
ldaOut.terms <- as.matrix(terms(ldaOut,100))
write.csv(ldaOut.terms,file=paste("LDAGibbs",k,"TopicsToTerms.csv"))


######关于LDA的另一种代码输入
library(quantreg)

set.seed(10)
m = LDA(articleDtm, method = "Gibbs", k = 6,  control = list(alpha = 0.1))
m

#输出各个topic频率最高的词
terms(m,5)

#LDA可视化结果
library(LDAvis)   

articleDtm = articleDtm[slam::row_sums(articleDtm) > 0, ]
phi = as.matrix(posterior(m)$terms)
theta <- as.matrix(posterior(m)$topics)
vocab <- colnames(phi)
doc.length = slam::row_sums(articleDtm)
term.freq = slam::col_sums(articleDtm)[match(vocab, colnames(articleDtm))]

json = createJSON(phi = phi, theta = theta, vocab = vocab,
                  doc.length = doc.length, term.frequency = term.freq)
serVis(json)
