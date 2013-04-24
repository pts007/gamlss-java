dat <- na.omit(read.csv("C:/Users/Daniil/Desktop/Gamlss_exp/data.csv", header = TRUE))
library(survival)

mod=gamlss(y~dat$x, 
           sigma.formula=~dat$x,
           nu.formula=~dat$x,
           tau.formula=~dat$x,
           family=BCPE, data=dat)

mod=gamlss(y~pb(dat$x), 
           sigma.formula=~pb(dat$x),
           nu.formula=~pb(dat$x),
           tau.formula=~x,
           family=BCPE, data=dat)


mod2=gamlss(y~rw(dat$y),  
            sigma.fo=~rw(dat$y), 
            nu.fo=~rw(dat$y),
            tau.fo=~rw(dat$y),
            data = dat, family=BCPE)

outR = na.omit(read.csv("C:/Users/Daniil/Desktop/Gamlss_exp/outR.csv", header = T))
outJ = na.omit(read.csv("C:/Users/Daniil/Desktop/Gamlss_exp/outJ.csv", header = F))


R2mu = 1-((sum((outJ$V1-outR$mu)^2)/length(dat$y))
        /(sum(((outR$mu)- mean(outR$mu))^2)/length(dat$y)))

R2sigma = 1-((sum((outJ$V2-outR$sigma)^2)/length(dat$y))
        /(sum(((outR$sigma)- mean(outR$sigma))^2)/length(dat$y)))

R2nu = 1-((sum((outJ$V3-outR$nu)^2)/length(dat$y))
        /(sum(((outR$nu)- mean(outR$nu))^2)/length(dat$y)))

R2tau = 1-((sum((outJ$V4-outR$tau)^2)/length(dat$y))
        /(sum(((outR$tau)- mean(outR$tau))^2)/length(dat$y)))

R2mu
R2sigma
R2nu
R2tau