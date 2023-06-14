# TD 2

rm(list=ls()); library(mclust)
K <- 3

# Données
data <- read.table(file='dryad_zebra.csv', header=TRUE, sep=',')
head(data)
plot(data$x, data$y, type='b', pch=20)
Y <- as.matrix(data[, c('speed', 'angle')])
n <- nrow(Y)

# Fonctions
Mstep <- function(Y, eStep){
  # eStep = list(tau = matrix(n * K), eta = array(n, K, K)))
  # nu : K * 1, pi : K * K
  # mu : K * 2, sigma : K, 2, 2
  nu <- eStep$tau[1, ]
  pi <- apply(eStep$eta, c(2, 3), sum)
  pi <- pi / rowSums(pi)
  N <- colSums(eStep$tau)
  mu <- matrix(NA, K, 2); sigma <- array(NA, dim=c(K, 2, 2))
  for(k in 1:K){
    mu[k, ] <- t(eStep$tau[, k])%*%Y / N[k]
    sigma[k, , ] <- (t(Y) %*% diag(eStep$tau[, k]) %*% Y) / N[k] - (mu[k, ]%o% mu[k, ])
  }
 return(list(nu=nu, pi=pi, mu=mu, sigma=sigma)) 
}
Init <- function(Y, K){
  # Modèle de mélange gaussien (GMM)
  n <- nrow(Y)
  gmm <- Mclust(Y, G=K)
  tau <- gmm$z
  eta <- array(0, dim=c(n, K, K))
  for(t in 2:n){eta[t, , ] <- tau[t-1, ]%o%tau[t, ]}
  return(list(tau=tau, eta=eta, logLik=gmm$loglik))
}
Forward <- function(Y, parms){
  n <- nrow(Y); K <- length(parms$nu)
  # Densités d'émission
  phi <- matrix(NA, n, K)
  for(k in 1:K){
    phi[, k] <- dmvnorm(Y, mean=parms$mu[k, ], sigma=parms$sigma[k, , ], log=TRUE)
  }
  phi[which(phi < -100)] <- -100; phi <- exp(phi)
  # Récurrence avant
  Fw <- matrix(NA, n, K)
  Fw[1, ] <- parms$nu * phi[1, ]
  logLik <- log(sum(Fw[1, ]))
  Fw[1, ] <- Fw[1, ] / sum(Fw[1, ])
  for(t in 2:n){
    Fw[t, ] <- Fw[t-1, ] %*% parms$pi
    Fw[t, ] <- Fw[t, ]* phi[t, ]
    logLik <- logLik + log(sum(Fw[t, ]))
    Fw[t, ] <- Fw[t, ] / sum(Fw[t, ])
  }
  return(list(phi=phi, Fw=Fw, logLik=logLik))
}
Backward <- function(parms, forward){
  n <- nrow(Y); K <- length(parms$nu)
  # Récurrence arrière
  tau <- G <- matrix(NA, n, K)
  eta <- array(0, dim=c(n, K, K))
  tau[n, ] <- forward$Fw[n, ]
  for(t in (n-1):1){
    G[t+1, ] <- forward$Fw[t, ] %*% parms$pi
    eta[t, , ] <- parms$pi * (forward$Fw[t, ] %o% (tau[t+1, ]/G[t+1, ]))
    tau[t, ] <- rowSums(eta[t, , ])
  }
  return(list(tau=tau, eta=eta))
}
HMM <- function(Y, K){
  
}

# Algo EM
eStep <- Init(Y, K)
iter <- 0; iterMax <- 1e2
tol <- 1e-6; diff <- 2*tol
logL <- rep(NA, iterMax)
while((diff > tol) & (iter < iterMax)){
  iter <- iter+1
  # Etape M
  parms <- Mstep(Y, eStep)
  # Etape E
  forward <- Forward(Y, parms)
  eStepNew <- Backward(parms, forward)
  # Test
  diff <- max(abs(eStepNew$tau - eStep$tau))
  eStep <- eStepNew
  logL[iter] <- forward$logLik
}
plot(logL, type='b', pch=20)

Zhat <- apply(eStep$tau, 1, which.max)
plot(data$x, data$y, type='b', pch=20, col=Zhat)
boxplot(Y[, 1] ~ Zhat)
boxplot(Y[, 2] ~ Zhat)
