put:
  rsync -avzhi --delete --filter=':- .gitignore' ./ ic-hpc:/rds/general/user/jws424/home/repos/neurips/trellis
