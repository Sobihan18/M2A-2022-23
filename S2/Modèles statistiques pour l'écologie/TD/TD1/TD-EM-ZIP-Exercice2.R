rm(list=ls())

# Données
tab <- read.table('BarentsFish.csv', header=TRUE, sep=';')
j <- 20; y <- as.vector(tab[, 4+j]); 
y0 <- 1*(y==0); n <- length(y)
x <- as.matrix(cbind(rep(1, n), tab[, 1:4]))

# Initialisation
alpha <- glm(y0 ~ -1 + x, family='binomial')$coef
beta <- glm(y ~ -1 + x, family='poisson')$coef

# Fonctions
objAlpha <- function(alpha, y0, x, tau){
  (t(tau)%*%x%*%alpha - sum(log(1+exp(x%*%alpha))))[1, 1]
}
gradAlpha <- function(alpha, y0, x, tau){
  as.vector(t(x)%*%(tau-plogis(x%*%alpha)))
}
objBeta <- function(beta, y, x, tau){
  (-sum((1-tau)*exp(x%*%beta)) + ((1-tau)*y)%*%x%*%beta)[1, 1]
}
gradBeta <- function(beta, y, x, tau){
  as.vector(t(x)%*%((1-tau)*(y - exp(x%*%beta))))
}
logLik <- function(alpha, beta, y, y0, x){
  pi <- plogis(x%*%alpha); lambda <- exp(x%*%beta)
  sum(log(pi*y0 + (1-pi)*dpois(y, lambda)))
}

# Algorithme EM
tol <- 1e-6; iterMax <- 1e3
diff <- 2*tol; iter <- 0
logL <- rep(NA, iterMax)
while((diff > tol) & (iter < iterMax)){
  iter <- iter+1
  # Etape E
  pi <- plogis(x%*%alpha); lambda <- exp(x%*%beta)
  tau <- as.vector(pi*y0 / (pi + (1-pi)*exp(-lambda)))
  
  # Etape M
  alphaNew <- optim(par=alpha, fn=objAlpha, gr=gradAlpha, y0=y0, x=x, tau=tau, 
                    control=list(fnscale=-1))$par
  betaNew <- optim(par=beta, fn=objBeta, gr=gradBeta, y=y, x=x, tau=tau, 
                   control=list(fnscale=-1))$par
  
  # Test & mise à jour
  diff <- max(abs(c(alpha, beta) - c(alphaNew, betaNew)))
  logL[iter] <- logLik(alpha, beta, y, y0, x)
  cat(iter, alpha, beta, logL[iter], diff, '\n')
  alpha <- alphaNew; beta <- betaNew
}
plot(logL[1:iter], pch=20, type='b')