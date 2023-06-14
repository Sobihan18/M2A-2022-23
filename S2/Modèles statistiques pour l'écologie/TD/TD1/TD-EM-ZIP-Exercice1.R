rm(list=ls())

# Données
tab <- read.table('BarentsFish.csv', header=TRUE, sep=';')
j <- 20; y <- tab[, 4+j]; y0 <- 1*(y==0)

# Initialisation
pi <- mean(y0)
lambda <- mean(y)

# Fonction
logLik <- function(pi, lambda, y, y0){
  sum(log(pi*y0 + (1-pi)*dpois(y, lambda)))
}

# Algorithme EM
tol <- 1e-6; iterMax <- 1e2
diff <- 2*tol; iter <- 0
logL <- rep(NA, iterMax)
while((diff > tol) & (iter < iterMax)){
  iter <- iter+1
  # Etape E
  tau <- pi*y0 / (pi + (1-pi)*exp(-lambda))
  
  # Etape M
  piNew <- mean(tau)
  lambdaNew <- sum((1-tau)*y) / sum(1-tau)
  
  # Test & mise à jour
  diff <- max(abs(c(pi, lambda) - c(piNew, lambdaNew)))
  logL[iter] <- logLik(pi, lambda, y, y0)
  cat(iter, pi, lambda, logL[iter], diff, '\n')
  pi <- piNew; lambda <- lambdaNew
}
plot(logL[1:iter], pch=20, type='b')

# Test du rapport de vraisemblance
logL0 <- sum(dpois(y, mean(y), log=TRUE))
logL1 <- logLik(pi, lambda, y, y0)
LRT <- 2*(logL1 - logL0)
pvalue <- pchisq(LRT, df=1, lower.tail=FALSE)/2
