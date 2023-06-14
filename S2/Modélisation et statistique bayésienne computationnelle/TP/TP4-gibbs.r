###########################################################################################################
#
#
#       APPLICATION  D'UN ALGORITHME DE GIBBS POUR UNE SIMULATION BAYESIENNE A POSTERIORI
#
#
#
#
#                  Donnees X  ~ Loi NORMALE (THETA,1). La derniere donnee observee est
#                                                      censuree à droite.
#
#                  A priori sur THETA = Loi NORMALE (MU,1)
#                                             d'hyperparametre MU 
#
#
#
#    L'algorithme de Gibbs simule des observations manquantes puis utilise la
#    conjugaison naturelle qui apparaît
#
#
############################################################################################################
options(warn=(-1))

#===============================================================================#
#
#      JEU DE DONNEES (15 DONNEES simulees selon la loi NORMALE (0,1) )
#         la derniere donneee est censuree à droite
#
#===============================================================================#

donnees <- c( 0.8114032, -0.4631576, -0.5984133,  1.5006175,  0.2123149, -1.0101033,1.0199253, -2.5994554, -0.3684774,  0.8263161,  0.4572680, -0.7240380,-0.3134977, -0.3772193, 1.64)




#===============================================================================#
#
#              Echantillonneur de Gibbs
#
#         N         = nombre de tirages desires a posteriori
#         mu        = moyenne a priori pour X 
#         nb.chains = nombre de chaînes de Gibbs //
#
#
#
#
#===============================================================================#

gibbs.sampling <- function(N=5000,nb.chains=3,mu=-1)
{

  # informations sur les donnees
  n <- length(donnees)
 
  # log-densite de la loi a priori sur theta
  log.prior <- function(theta)
                 {
                  return(- (1/2)*log(2*pi) - (1/2)*(theta-mu)^2)
                 } 
  prior.dens  <- function(x){exp(log.prior(x))}

  # log-densite de la loi a posteriori sur theta definie à un facteur pres
  moy.post   <- (mu + sum(donnees[1:(n-1)]))/n 
  sigma.post <-  sqrt(1/n)
  log.post <- function(theta)
                 {
                  ll1 <- - (1/(2*sigma.post^2))*(theta-moy.post)^2              # terme gaussien regulier
                  ll2 <- log(1-pnorm((donnees[n]-theta),0,1))                   # terme dû à la censure
                  res <- ll1 + ll2
                  return(res)
                 }
   
  
  # Fonction quantile de la loi normale tronquee (G. Pujol, B. Iooss)
  qtnorm <- function(p, mean = 0, sd = 1, min = -1e6, max = 1e6){return(qnorm((1 - p) * pnorm(min, mean, sd) + p * pnorm(max, mean, sd),mean, sd))}

  # Fonction de generation de variables aleatoires de la loi normale tronquee (G. Pujol, B. Iooss)
  rtnorm <- function(n, mean = 0, sd = 1, min = -1e6, max = 1e6){return(qtnorm(runif(n), mean, sd, min, max))}


 
  # ---------------- BOUCLE DE SIMULATION --------------------------------
  NN          <- 20*N                                                           # Heuristique de Rubin
  theta.cur   <- theta.samp <- rnorm(nb.chains,mu,1)                            # initialisation des chaînes  (loi a priori)
  k = 0
  GB.stat <- c()
  layout(matrix(c(1:2),1,2))                                                    # parametres graphiques
  layout.show(2)
  while (TRUE)
  {
   k = k+1
   
   # Simulation de la donnee manquante (loi normale tronquee par la valeur de censure)
   troncature <- donnees[n]
   donnee.manquante <- rtnorm(nb.chains,mean=theta.cur,sd=1,min=troncature)
  
   # Simulation de la nouvelle distribution courante de theta
   donnees.cur <- cbind(t(matrix(donnees[1:(n-1)],(n-1),nb.chains)),donnee.manquante)   # matrice de taille nb.chains x n 
   moyenne     <- (mu + apply(donnees.cur,1,sum,na.rm=TRUE))/(n+1)                      # vecteur de taille nb.chains
   ec.type     <- sqrt(1/(n+1))
   theta.prov  <- rnorm(NN,moyenne,ec.type)
   theta.cur   <- theta.prov[1:nb.chains]                                               # remise à jour des chaînes  
   theta.samp  <- cbind(theta.samp,theta.cur)
  
  
   #-------------- Graphiques dynamiques 
   # Premiere fenêtre : calcul de la statistique de Brooks-Gelman
   K=10               # nombre d'iteration avant le premier calcul
   GB.stat.prov <- NA
   if (k>K){GB.stat.prov <- Brooks.Gelman(theta.samp,k,nb.chains=nb.chains)}
   GB.stat <- c(GB.stat,GB.stat.prov)
   plot(c(1:k),GB.stat,"l",lwd=2,ylim=c(0.5,3),xlab="nb.iterations",ylab="",main="Statistique de Brooks-Gelman (theta)")
  
   # Seconde fenêtre
   histo=hist(theta.prov,min(c(40,sqrt(length(theta.prov)))),col=gray(0.7),freq=F,xlim=c(-2,2),main="Histogramme des simulations",xlab="theta",ylab="densite")
   curve(prior.dens(x),add=T,col=2,lty=3,lwd=2)
   points(donnees[1:(n-1)],rep(0,(n-1)),col=6,pch=4,lwd=3)
   points(donnees[n],0,col=4,pch=1,lwd=3)
   yo.max = max(histo$density)
   legend(0.6,yo.max,c("a priori","a posteriori"),col=c(2,2),lty=c(3,1),lwd=rep(2,2),bty="n",cex=1.1)
   legend(0.7,(yo.max-0.3),c("donnees iid","censure"),col=c(6,4),pch=c(4,1),bty="n",cex=1.1,lwd=c(2,2)) 
  
   
   if ((k>K)&(sum((GB.stat<1.025),na.rm=TRUE)>100)){break}
    
  }
  
 # reechantillonnage avec des poids equitables 
 theta.post = sample(theta.prov,N)
 
 
 
 # estimation de la constante de normalisation par integration numerique et validation graphique de la simulation obtenue
 f <- function(x){exp(log.post(x))}
 inv.constante.C  <- integrate(f,lower=-Inf,upper=Inf)$value
 constante.C      <- 1/(inv.constante.C) 
 post.dens   <- function(x){constante.C*exp(log.post(x))}
 curve(post.dens(x),add=T,col=2,lwd=2,lty=1)


 #return(theta.post[1:N])
} 




#===============================================================================#
#
#                 Statistique de Brooks-Gelman
#                 calculee sur un nombre nb.chains
#                 de chaines MCMC paralleles pour
#                 le parametre theta (matrix nbchains X Nsim)
#
#
#===============================================================================#

Brooks.Gelman  <- function(theta, Nsim, nb.chains=3, pro=0.9)
{
delta <- rep(0,nb.chains)
      
      
      
for(j in 1:nb.chains)
    {
    delta[j] <- (quantile(theta[j,(floor(Nsim/2)):Nsim], probs=pro,na.rm=TRUE)-quantile(theta[j,(floor(Nsim/2)):Nsim], probs=1-pro,na.rm=TRUE))
    }

delta.bas <- sum(delta,na.rm=TRUE)/nb.chains
del <- (quantile(theta[,(floor(Nsim/2)):Nsim], probs=pro,na.rm=TRUE)-quantile(theta[,(floor(Nsim/2)):Nsim], probs=1-pro,na.rm=TRUE))

R <- del/delta.bas
return(R)
}


