
library(sp)
library(raster)
library(rgdal)

setwd('/Users/jiesun/Dropbox/PROJECT_JE/Generate_Cases')

# 1. Assign_Endemic_Regions
print('Step 1: pre-ran. Output: Coord_Regions_Final.Rds, Country_Index.Rds')

# 2. Adjust_Pop_To_Match_UN
### NOTE ###
# Adjust Population Dataframe (Map) to match with Quan (orginal also from UN but adjusted for some subnation regions in RUS, PAK, AUS) and UN data
# CHN = CHN + MAC (as mapping MAC, idx = 14, is inside CHN)
# ratio = alternative / Duy --> adjusted pop = Duy * ratio
### ---- ###
cat('===== START [Adjust_Pop_To_Match_UN.R] =====\n')

print('Step 2: pre-ran. Output: Generate/Adjusted_Pop_Map.Rds')

cat('===== FINISH [Adjust_Pop_To_Match_UN.R] =====\n')

# 3. Extract_Age_Distribution_Population

### NOTE ###
# Convert format from Quan (30 regions) into countries (25 countries)
# The result will be age-distributed population in the year 2015 at each countries (total 25 countries)
### ---- ###

cat('===== START [Extract_Age_Distribution_Population.R] =====\n')

print('Step 3: pre-ran. Output:Naive_pop_24ende_2015_Country.Rds')

cat('===== FINISH [Extract_Age_Distribution_Population.R] =====\n')

# 4. Generate_Cases_Dataframe
# NOTE #
# Move to the last line of this file to get more information about which files that this script will save --> Default is to save all of agegroup and the total cases of all group --> 101 files
# Basically same with Generate_Cases_DF.R --> but here use both original VIMC data and Quan (VIMC+UN) data
# Create Cases per pixel map based on FOI map and Population VIMC file
# Use catalytic model to find cases: Cases = [1 - exp(-lambda)] * exp(-lambda * age) * symtomatic_rate * pop_age
# lambda: FOI from map
# age: age of the group want to find cases (0, 1, 2, ..., 99)
# symtomatic_rate: rate from sampling 
# pop_age: population at the specific age --> find by using VIMC pop file --> age distribution --> pop per pixel map
# Setting with (seed = 114) --> this is used to generate cases for VIMC last year
# PSym <- runif(1600, 1/500, 1/250)
# PMor <- runif(1600, 0.1, 0.3) 
# PDis <- runif(1600, 0.3, 0.5)
# However, Quan did stored his parameters for these values -->  we can used this to compared with Quan
# ---- #

cat('===== START [Generate_Cases_Dataframe.R] =====\n')
cat('===== Model RF ======') #@@
m_name = 'RF' #@@

# # Get directory of the script (this part only work if source the code, wont work if run directly in the console)
# # This can be set manually !!!
# script.dir <- dirname(sys.frame(1)$ofile)
# script.dir <- paste0(script.dir, '/')
# setwd(script.dir)
# # Create folder to store the generate cases result (will show warnings if the folder already exists --> but just warning, no problem)
dir.create(file.path('Generate/Cases/RF/'), showWarnings = TRUE)

####### Generate Cases Pixel DF #######
# Read FOI Map file
df.csv <- read.csv('Data/Endemic_FOI_RF_Quantile_Regression_Full_Cov_400.csv', sep = '\t') #@@place to change file
df.csv <- df.csv[, -1]
df.foi <- df.csv[, c(1, 2, 3)] 
rm(df.csv)

##### Read Pop Map Data (Dataframe training) #####
df.pop <- readRDS('Generate/Adjusted_Pop_Map.Rds')
colnames(df.pop) <- c('x', 'y', 'Pop')

##### Read Population data collected by Quan (UN and subnational regions in PAK, AUS, RUS) #####
# This file is the result after running Extract_Age_Distribution_Population.R
vimc.pop <- readRDS('Generate/Naive_pop_24ende_2015_Country.Rds')

##### Read assigned regions dataframe #####
# Note that MACAU is 14 but there is no pixel is assigned as 14 since MACAU belongs to China (index 5)
df.regions <- readRDS('Generate/Coord_Regions_Final.Rds') 
Country_Idx <- readRDS('Generate/Country_Index.Rds')

##### Set up symptomatic rate #####
# set.seed(114) # make sure the sampling is the same all the time we run the file
# PSym <- runif(1600, 1/500, 1/250)
# PSym_mean <- mean(PSym) # take means

# take same value with Quan in order to make it easy for comparing
# PSym <- readRDS('JE_model_Quan/data/uncertainty_quantities/symp_rate_dist.rds')
PSym <- readRDS('Data/symp_rate_dist.rds')
PSym_mean <- mean(PSym)

##### Find age distribution of each country supported by VIMC #####
vimc.pop$distribution <- 0
countries <- unique(vimc.pop$country_code) # countries which are supported by VIMC
for (country in countries){
  idx_row <- which(vimc.pop$country_code == country)
  pop_sum <- sum(vimc.pop$X2015[idx_row])
  pop_distribute <- vimc.pop$X2015[idx_row] / pop_sum
  vimc.pop$distribution[idx_row] <- pop_distribute
}

###### Create empty dataframe for age pop distribution (nrow = npixel, ncol = 100 agegroup + 2 coord) #####
headers <- c('x', 'y', paste0('Age_0', c(0:9)), paste0('Age_', c(10:99)))
df.popage <- as.data.frame(matrix(0, ncol = 102, nrow = nrow(df.foi)))
colnames(df.popage) <- headers
df.popage$x <- df.foi$x
df.popage$y <- df.foi$y

for (i in 1 : length(Country_Idx)){
  country <- Country_Idx[i]
  cat('Processing', country, '\n')
  if (country != 'MAC'){ # Do not run for MAC
    idx_region <- which(df.regions$regions == i)
    if (country == 'Low.NPL' || country == 'High.NPL'){
      country <- 'NPL'
    }
    if (country == 'HKG'){ # Used age distribution of CHN and apply to HKG
      country <- 'CHN'
    }
    idx_vimc_pop <- which(vimc.pop$country_code == country)
    if (length(idx_vimc_pop) == 0){ # DO NOT HAVE VIMC POP DATA FOR THIS COUNTRY
      for (idx_col in 3 : ncol(df.popage)){
        df.popage[[idx_col]][idx_region] <- df.pop$Pop[idx_region] / 100 # equally portion for all 100 age groups
      }
    }else{ # HAVE VIMC POP DATA
      for (idx_col in 3 : ncol(df.popage)){
        df.popage[[idx_col]][idx_region] <- df.pop$Pop[idx_region] * vimc.pop$distribution[idx_vimc_pop[idx_col - 2]]
      }
    }
  }
}

###### Find Cases for each pixels for each age group #####
rm(df.pop, df.regions, vimc.pop)
headers <- c('x', 'y', paste0('Age_0', c(0:9)), paste0('Age_', c(10:99)))
df.casesage <- as.data.frame(matrix(0, ncol = 102, nrow = nrow(df.foi)))
colnames(df.casesage) <- headers
df.casesage$x <- df.foi$x
df.casesage$y <- df.foi$y
for (i in 3 :  ncol(df.popage)){
  df.casesage[[i]] <- (1 - exp(-1 * df.foi$Predict)) * exp(-1 * df.foi$Predict * (i - 3)) * PSym_mean * df.popage[[i]]
}
df.casesage$Total <- rowSums(df.casesage[,3:102])
# saveRDS(df.casesage, 'Cases_DF_Agegroup.Rds') # intensive file, large size --> dont recommend to save this file

# Save entire cases at each age group
# If only want to save the total cases of all agegroup --> set i = ncol(df.casesage) --> remove the loop
for (i in 3 : ncol(df.casesage)){
  # i <- ncol(df.casesage)
  cat('Saving', colnames(df.casesage)[i], '\n')
  df <- df.casesage[ ,c(1, 2, i)]
  saveRDS(df, paste0('Generate/Cases/RF/Cases_', colnames(df.casesage)[i],m_name, '.Rds')) #@@
}

cat('===== FINISH [Generate_Cases_Dataframe.R] =====\n')

# 5. Generate_Cases_Map
# NOTE #
# Generate Map (TIF) files from dataframe (From Generate_Cases_Dataframe.R)
# Move to the last line of this file to get more information about which files that this script will save
# Default is only plot the total cases of all agegroup --> take the last file in the folder
# ---- #


cat('===== START [Generate_Cases_Map.R] =====\n')

# # Get directory of the script (this part only work if source the code, wont work if run directly in the console)
# # This can be set manually !!!
# script.dir <- dirname(sys.frame(1)$ofile)
# script.dir <- paste0(script.dir, '/')
# setwd(script.dir)
# # Create folder to store the generated raster result (will show warnings if the folder already exists --> but just warning, no problem)
dir.create(file.path('Generate/Cases_TIF/RF/'), showWarnings = TRUE) #@@

# SJ: the below code is problematic
create_raster_from_df <- function(dataframe, res = c(5, 5),
                                  crs = "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=km +no_defs",
                                  name = 'rasterdf', savefile = FALSE){
    crs <- crs(crs)
    rasterdf <- rasterFromXYZ(dataframe, res = res, crs = crs)
    if (savefile){
        writeRaster(rasterdf, name, overwrite = TRUE, format = "GTiff")
    }
    return(rasterdf)
}

# SJ: tried adjusted below. But recursion is too deep
# crs<- function(dataframe, res = c(5, 5),
#                crs = "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=km +no_defs",
#                name = 'rasterdf', savefile = FALSE){
#   crs <- crs(crs)
#   rasterdf <- rasterFromXYZ(dataframe, res = res, crs = crs)
#   if (savefile){
#     writeRaster(rasterdf, name, overwrite = TRUE, format = "GTiff")
#   }
#   return(rasterdf)
# }
crs_new <- crs("+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=km +no_defs")

# Read data
# These data is the result after running Generate_Cases_Dataframe.R
LinkData <- 'Generate/Cases/RF/'
ListFiles <- list.files(LinkData)

# Create Map (loop)
# for (i in 1 : length(ListFiles)){
i <- length(ListFiles) # to generate the total cases of all agegroup
df <- readRDS(paste0(LinkData, ListFiles[i]))
rasterdf <- rasterFromXYZ(df, res = c(5, 5), crs = crs_new)
Namefile = paste0('Generate/Cases_TIF/RF/Cases_', colnames(df)[3], m_name) #@@
writeRaster(rasterdf, Namefile, overwrite = TRUE, format = "GTiff")
# }

cat('===== FINISH [Generate_Cases_Map.R] =====\n')


# 6. Generate_Cases_Map_Country

# NOTE #
# Generate Map (SHP) files for each country (total cases in a country) from dataframe (From Generate_Cases_Dataframe.R)
# Compare to VIMC Result (Modelling way)
# ---- #


cat('===== START [Generate_Cases_Map_Country.R] =====\n')

# # Get directory of the script (this part only work if source the code, wont work if run directly in the console)
# # This can be set manually !!!
# script.dir <- dirname(sys.frame(1)$ofile)
# script.dir <- paste0(script.dir, '/')
# setwd(script.dir)
# # Create folder to store the generated SHP result (will show warnings if the folder already exists --> but just warning, no problem)
dir.create(file.path('Generate/Cases_SHP/RF'), showWarnings = TRUE) #@@


##### CREATE SHP FILE #####

# SJ: the below code is problematic
create_raster_from_df <- function(dataframe, res = c(5, 5),
                                  crs = "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=km +no_defs",
                                  name = 'rasterdf', savefile = FALSE){
    crs <- crs(crs)
    rasterdf <- rasterFromXYZ(dataframe, res = res, crs = crs)
    if (savefile){
        writeRaster(rasterdf, name, overwrite = TRUE, format = "GTiff")
    }
    return(rasterdf)
}

# crs <- function(dataframe, res = c(5, 5),
#                 crs = "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=km +no_defs",
#                 name = 'rasterdf', savefile = FALSE){
#   crs <- crs(crs)
#   rasterdf <- rasterFromXYZ(dataframe, res = res, crs = crs)
#   if (savefile){
#     writeRaster(rasterdf, name, overwrite = TRUE, format = "GTiff")
#   }
#   return(rasterdf)
# }
crs_new <- crs("+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=km +no_defs")

# Read data regions DF
df.regions <- readRDS('Generate/Coord_Regions_Final.Rds')

# Read SHP File of endemic area
region.shp <- readOGR('Data/Shapefile_Endemic/Ende_map_feed.shp')
countries <- region.shp@data$Country
countries <- as.character(countries) # countries in endemic areas

# Read data
LinkData <- 'Generate/Cases/RF/'
ListFiles <- list.files(LinkData)
idx_file <- length(ListFiles) # Total cases of all age group
df <- readRDS(paste0(LinkData, ListFiles[idx_file]))

region.shp@data$Cases <- 0
# Find Cases for each country
for (idx_country in 1 : length(countries)){
  country <- countries[idx_country]
  if (country != 'MAC'){
    idx_row_regions <- which(df.regions$regions == idx_country)
    total_cases <- sum(df$Total[idx_row_regions])
    region.shp@data$Cases[idx_country] <- total_cases    
  }
}

writeOGR(region.shp, ".", "Generate/Cases_SHP/RF/Total_Cases_SHP", driver="ESRI Shapefile") #@@ SJ: it's not saving to the Cases_SHP folder

##### COMPARE WITH VIMC: Comparing with the result of VIMC templates we ran last year #####
# vimc <- read.csv("~/DuyNguyen/RProjects/Rstan_Quan/Template_Generate/2017gavi6/MeanCases/naive_2017gavi6.csv")
# vimc.2015 <- vimc[which(vimc$year == 2015), ]
# rm(vimc)
# vimc.2015 <- vimc.2015[, c(3, 4, 8)] # Age, Country, Cases
# vimc.2015$country <- as.character(vimc.2015$country)
# 
# countries <- unique(vimc.2015$country)
# vimc.cases <- data.frame(country = countries, cases = 0)
# vimc.cases$country <- as.character(vimc.cases$country)
# 
# for (idx.country in 1 : length(countries)){
#     country <- countries[idx.country]
#     cases <- sum(vimc.2015$cases[which(vimc.2015$country == country)])
#     vimc.cases$cases[idx.country] <- cases
# }
# 
# rf <- readOGR('~/DuyNguyen/RProjects/OUCRU JE/Generate_Case_Map/Cases_Country_SHP/Cases_SHP.shp')
# rfdata <- rf@data
# rm(rf)
# rfdata <- rfdata[, c(1, 4)] # Country, Cases
# rfdata$Country <- as.character(rfdata$Country)
# # Fix problem Low.NPL + High.NPL
# idx_low <- which(rfdata$Country == 'Low.NPL')
# idx_high <- which(rfdata$Country == 'High.NPL')
# rfdata$Cases[idx_low] <- rfdata$Cases[idx_low] + rfdata$Cases[idx_high]
# rfdata$Country[idx_low] <- 'NPL'
# rfdata <- rfdata[-idx_high, ]
# 
# # Compare 2 data #
# compare <- data.frame(country = countries, vimc = vimc.cases$cases, rf = 0)
# for (idx.country in 1 : length(countries)){
#     country <- countries[idx.country]
#     idx <- which(rfdata$Country == country)
#     compare$rf[idx.country] <- rfdata$Cases[idx]
# }

cat('===== FINISH [Generate_Cases_Map_Country.R] =====\n')
