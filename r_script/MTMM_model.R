### THIS SCRIPT CONTAINS CODES TO PREFORM MTMM MODEL

# BASIC MTMM-1 MODEL
library(lavaan)
mtmm_basic <- "
  T1_uni =~ 1*universalism + uni_xgb + uni_rf
  T2_tra =~ 1*tradition + tra_xgb + tra_rf
  T3_pow =~ 1*power + pow_xgb + pow_rf

  M1_survey =~ 1*universalism + 1*tradition + 1*power
  M2_fb_rf =~ 1*uni_rf + 1*tra_rf + 1*pow_rf
  M3_fb_xgb =~ 1*uni_xgb + 1*tra_xgb + 1*pow_xgb
  #M4_fb_nn =~ 1*uni_nn + 1*tra_nn + 1*pow_nn

  #T1_uni~~1*T1_uni
  #T2_tra~~1*T2_tra
  #T3_pow~~1*T3_pow

  T1_uni~~T2_tra + T3_pow 
  T2_tra~~T3_pow
"
fit_mtmm_basic <- lavaan(mtmm_basic, data = full_data, missing = "ml", 
                         auto.fix.first = TRUE, auto.var = TRUE, auto.cov.lv.x = FALSE)
summary(fit_mtmm_basic, standardized = TRUE, fit.measures = TRUE)


# TRUE SCORE MTMM-1 MODEL
mtmm_ts <- "
  # the reliability
  universalism_TS =~ 1*universalism
  tradition_TS =~ 1*tradition
  power_TS =~ 1*power
  uni_rf_TS =~ 1*uni_rf
  tra_rf_TS =~ 1*tra_rf
  pow_rf_TS =~ 1*pow_rf
  uni_xgb_TS =~ 1*uni_xgb
  tra_xgb_TS =~ 1*tra_xgb
  pow_xgb_TS =~ 1*pow_xgb
  
  # the true score validity
  T1_uni =~ 1*universalism_TS + uni_rf_TS + uni_xgb_TS
  T2_tra =~ 1*tradition_TS + tra_rf_TS + tra_xgb_TS
  T3_pow =~ 1*power_TS + pow_rf_TS + pow_xgb_TS

  M1_survey =~ 1*universalism_TS + 1*tradition_TS + 1*power_TS
  M2_fb_rf =~ 1*uni_rf_TS + 1*tra_rf_TS + 1*pow_rf_TS
  M3_fb_xgb =~ 1*uni_xgb_TS + 1*tra_xgb_TS + 1*pow_xgb_TS

  #universalism ~~ universalism
  #tradition ~~ tradition
  #power ~~ power
  #uni_rf ~~ uni_rf
  #tra_rf ~~ tra_rf
  #pow_rf ~~ pow_rf

  #M1_survey ~~ M1_survey
  #M2_fb_rf ~~ M2_fb_rf
  #M3_fb_xgb ~~ M3_fb_xgb

  #T1_uni ~~ 1*T1_uni
  #T2_tra ~~ 1*T2_tra
  #T3_pow ~~ 1*T3_pow

  T1_uni ~~ T2_tra+T3_pow 
  T2_tra ~~ T3_pow
  
  universalism_TS ~~ universalism_TS 
  tradition_TS ~~ tradition_TS
  power_TS ~~ power_TS
  uni_rf_TS ~~ uni_rf_TS
  tra_rf_TS ~~ tra_rf_TS
  pow_rf_TS ~~ pow_rf_TS
  uni_xgb_TS ~~ uni_xgb_TS
  tra_xgb_TS ~~ tra_xgb_TS
  pow_xgb_TS ~~ pow_xgb_TS

"

fit_mtmm_ts <- lavaan(mtmm_ts, data = full_data, missing = "ml", 
                      auto.fix.first = FALSE, auto.var = FALSE)
summary(fit_mtmm_ts, standardized = TRUE, fit.measures = TRUE)
# the true score validity and reliability are pretty high


# Calculating standardized coefficients in one model from the other
std_ts <- standardizedsolution(fit_mtmm_ts) %>% filter(op == "=~") %>% select(1:4)
std_ts
std_basic <- standardizedsolution(fit_mtmm_basic) %>% 
  filter(op == "=~") %>% select(1:4)
std_basic

reliability_coefficient <- std_ts$est.std[std_ts$rhs == "universalism"]
val_and_met_coefficients <-  std_ts$est.std[std_ts$rhs == "universalism_TS"]
reliability_coefficient * val_and_met_coefficients
std_basic %>% filter(rhs == "universalism")

basic_coefs <- std_basic %>% filter(rhs == "universalism") %>% .$est.std
lambda <- basic_coefs[1]
gamma <- basic_coefs[2]
r <- sqrt(lambda^2 + gamma^2)
v <- lambda / r
m <- gamma / r
c(r=r, v=v, m=m)

inspect(fit_mtmm_basic, "cor.lv")
inspect(fit_mtmm_ts, "cor.lv")

# BASIC MTMM MODEL - WITH CORRELATION MATRIX
corr_3x3 <- cor(full_data[,c("universalism","tradition","power","uni_xgb","tra_xgb","pow_xgb","uni_rf","tra_rf","pow_rf")])
colnames(corr_3x3) <- rownames(corr_3x3) <- 
  paste0("Y", 1:9)

cov_3x3 <- cov(full_data[,c("universalism","tradition","power","uni_xgb","tra_xgb","pow_xgb","uni_rf","tra_rf","pow_rf")])
colnames(cov_3x3) <- rownames(cov_3x3) <- 
  paste0("Y", 1:9)

syn <- "
  T1 =~ 1*Y1 + Y4 + Y7
  T2 =~ 1*Y2 + Y5 + Y8
  T3 =~ 1*Y3 + Y6 + Y9
  
  M1 =~ 1*Y1 + 1*Y2 + 1*Y3
  M2 =~ 1*Y4 + 1*Y5 + 1*Y6
  M3 =~ 1*Y7 + 1*Y8 + 1*Y9
  
  T1 ~~ T2 + T3
  T2 ~~ T3
  M2 ~~ M3
"
fit <- lavaan(syn, sample.cov = corr_3x3, 
              sample.nobs = 374, 
              auto.cov.lv.x = FALSE, 
              auto.fix.first = TRUE, 
              auto.var = TRUE,
              estimator = 'ml')
summary(fit, standardized = TRUE, fit.measures = TRUE)


# TRUE SCORE MTMM MODEL - WITH CORRELATION MATRIX
syn_ts <- "
  Y1_ts =~ 1*Y1
  Y2_ts =~ 1*Y2
  Y3_ts =~ 1*Y3
  Y4_ts =~ 1*Y4
  Y5_ts =~ 1*Y5
  Y6_ts =~ 1*Y6
  Y7_ts =~ 1*Y7
  Y8_ts =~ 1*Y8
  Y9_ts =~ 1*Y9
  
  T1 =~ 1*Y1_ts + Y4_ts + Y7_ts
  T2 =~ 1*Y2_ts + Y5_ts + Y8_ts
  T3 =~ 1*Y3_ts + Y6_ts + Y9_ts
  
  M1 =~ 1*Y1_ts + 1*Y2_ts + 1*Y3_ts
  M2 =~ 1*Y4_ts + 1*Y5_ts + 1*Y6_ts
  M3 =~ 1*Y7_ts + 1*Y8_ts + 1*Y9_ts
  
  T1 ~~ T2 + T3
  T2 ~~ T3
  
  Y1_ts~~0*Y1_ts
  Y2_ts~~0*Y2_ts
  Y3_ts~~0*Y3_ts
  Y4_ts~~0*Y4_ts
  Y5_ts~~0*Y5_ts
  Y6_ts~~0*Y6_ts
  Y7_ts~~0*Y7_ts
  Y8_ts~~0*Y8_ts
  Y9_ts~~0*Y9_ts
"
fit_ts <- lavaan(syn_ts, sample.cov = corr_3x3, 
                 sample.nobs = 300, 
                 auto.cov.lv.x = FALSE, 
                 auto.fix.first = TRUE, 
                 auto.var = TRUE)
summary(fit_ts, standardized = TRUE, fit.measures = TRUE)

fit_ts_cov <- lavaan(syn_ts, sample.cov = cov_3x3, 
                 sample.nobs = 374, 
                 auto.cov.lv.x = FALSE, 
                 auto.fix.first = TRUE, 
                 auto.var = TRUE)
summary(fit_ts_cov, standardized = TRUE, fit.measures = TRUE)
