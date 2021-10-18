library(tm)
library(SnowballC)
library(topicmodels)
#设置工作目录（根据需要修改路径）
setwd("/Users/emma/Dropbox/Mac/Desktop/database3_1017") 
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

#correlations 找到单词之间的关联
#findAssocs(articleDtm,"green",0.8)
#findAssocs(articleDtm,"environment",0.6)
#findAssocs(articleDtm,"carbon",0.8)
#findAssocs(articleDtm,"purchasing",0.6)
#findAssocs(articleDtm,"warehousing",0.6)
#findAssocs(articleDtm,"delivery",0.8)
#findAssocs(articleDtm,"production",0.8)
#findAssocs(articleDtm,"transport",0.8)

#histogram 直方图
wf= data.frame(term=names(freq),occurrences=freq)
library(ggplot2)
p <- ggplot(subset(wf,freq>1000),aes(term,occurrences))
p <- p + geom_bar(stat="identity")
p <- p + theme(axis.text.x = element_text(angle = 45,hjust = 1))
p

#wordcloud 云图
library(wordcloud)
set.seed(1234)
#limit words by specifying min frequency
wordcloud(names(freq),freq,min.freq = 900)
wordcloud(names(freq),freq,min.freq = 900,color = brewer.pal(6,"Dark2"))

# Load topic models library
library(topicmodels)
library(quantreg)
library(quanteda)

set.seed(100)
m = LDA(articleDtm, method = "Gibbs", k = 6,  control = list(alpha = 0.1))
m

#输出各个topic频率最高的词
terms(m,5)


# create the visualization
library(tidytext)

ap_topics <- tidy(m, matrix = "beta")

library(ggplot2)
library(dplyr)

ap_top_terms <- ap_topics %>%
  group_by(topic) %>%
  top_n(20, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

ap_top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()
library(tidyr)

beta_spread <- ap_topics %>%
  mutate(topic = paste0("topic", topic)) %>%
  spread(topic, beta) %>%
  filter(topic1 > .001 | topic2 > .001) %>%
  mutate(log_ratio = log2(topic2 / topic1))

write.csv(beta_spread,"beta_spread.csv")




#########################################
##计算困惑度1 perplexity
library(quanteda) 
texts = corpus_reshape(data_corpus_inaugural, to = "paragraphs")
dfm = dfm(texts, remove_punct=T, remove=stopwords("english"))
dfm = dfm_trim(dfm, min_docfreq = 1000)

dtm = convert(dfm, to = "topicmodels") 

library(topicmodels)

train = sample(rownames(dtm), nrow(dtm) * .75)
dtm_train = dtm[rownames(dtm) %in% train, ]
dtm_test = dtm[!rownames(dtm) %in% train, ]

perplexity = data.frame(k = 2:6, p=NA)

for (k in perplexity$k) {
  message("k=", k)
  m1 = LDA(dtm_train, method = "Gibbs", k = k,  control = list(10, 10, 5))
  perplexity$p[perplexity$k==k] = perplexity(m1, dtm_test)
}
perplexity
plot(x=perplexity$k, y=perplexity$p)
######################################


# load up some R packages including a few we'll need later
library(topicmodels)
library(doParallel)
library(ggplot2)
library(scales)

data("articleDtm", package = "topicmodels")

burnin = 1000
iter = 1000
keep = 50

full_data  <- articleDtm
n <- nrow(full_data)
#-----------validation--------
k <- 5

splitter <- sample(1:n, round(n * 0.75))
train_set <- full_data[splitter, ]
valid_set <- full_data[-splitter, ]

fitted <- LDA(train_set, k = k, method = "Gibbs",
              control = list(burnin = burnin, iter = iter, keep = keep) )
perplexity(fitted, newdata = train_set) # about 2700
perplexity(fitted, newdata = valid_set) # about 4300



#困惑度可视化
library(ggplot2)
ggplot(p, aes(x=k, y=perplexity)) + geom_line()


# write out results
# docs to topics

m.topics <- as.matrix(topics(m))
write.csv(m.topics,file=paste("LDAGibbs",k,"DocsToTopics.csv"))

# Top N terms in each topic
m.terms <- as.matrix(terms(m,100))
write.csv(mt.terms,file=paste("LDAGibbs",k,"TopicsToTerms.csv"))

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


