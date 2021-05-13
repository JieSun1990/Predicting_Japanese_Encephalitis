# --- NOTE ---
# This script is a supported script (not really important and not complicated)
# This script is used to convert Rds (dataframe) to csv files (Because Python can read csv, cannot read Rds file)
# ---------- #

## Get directory of the script (this part only work if source the code, wont work if run directly in the console)
## This can be set manually !!! -->setwd('bla bla bla')
script.dir <- dirname(sys.frame(1)$ofile)
script.dir <- paste0(script.dir, '/')
setwd(script.dir)

## Create folder to store the result (will show warnings if the folder already exists --> but just warning, no problem)
dir.create(file.path('Generate/Python_CSV/'), showWarnings = TRUE)
Path_csv <- 'Generate/Python_CSV/'

# Example Use 1 --> Convert EM_Imputed_Features_Study.Rds to CSV
Path_Rds <- 'Generate/EM_DF/'
file_Rds <- 'EM_Imputed_Features_Study'
data <- readRDS(paste0(Path_Rds, file_Rds, '.Rds'))

file_csv <- paste0(file_Rds, '.csv')
write.csv(data, paste0(Path_csv, file_csv), row.names = FALSE)

# Example Use 2 --> Convert Imputed_Features_Endemic to CSV
Path_Rds <- 'Generate/Imputed_DF/'
file_Rds <- 'Imputed_Features_Endemic'
data <- readRDS(paste0(Path_Rds, file_Rds, '.Rds'))

file_csv <- paste0(file_Rds, '.csv')
write.csv(data, paste0(Path_csv, file_csv), row.names = FALSE)

# # Example Use 3 --> Sample small portion of EM_Imputed_Features_Study.Rds and Convert it to CSV
# set.seed(881994)
# Path_Rds <- 'Generate/EM_DF/'
# file_Rds <- 'EM_Imputed_Features_Study'
# data <- readRDS(paste0(Path_Rds, file_Rds, '.Rds'))
# data_small <- data[sample(1:nrow(data), 10000), ] # --> only sample 10000 pixels
# 
# file_csv <- paste0(file_Rds, '_Small.csv')
# write.csv(data_small, paste0(Path_csv, file_csv), row.names = FALSE)
