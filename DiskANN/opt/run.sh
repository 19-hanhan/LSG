DiskANN=/home/hanhan/Projects/diskann
opt=$DiskANN/opt

cd $opt

# ./prepare.sh [dataset]

# ./prepare.sh enron 0.01
# ./prepare.sh trevi 0.01
# ./prepare.sh notre 0.01
# ./prepare.sh millionSong 0.01
# ./prepare.sh random 0.01
# ./prepare.sh gaussian 0.01
# ./prepare.sh sift1M 0.01
# ./prepare.sh deep1M 0.01
# ./prepare.sh word2vec 0.01
# ./prepare.sh gist 0.01
# ./prepare.sh glove1M 0.01
# ./prepare.sh glove2M 0.01
# ./prepare.sh imageNet 0.01
# ./prepare.sh tiny5m 0.01
# ./prepare.sh deep10M 0.001
# ./prepare.sh sift10M 0.001
# ./prepare.sh spacev10M 0.001
# ./prepare.sh tiny80M 0.001

# ./prepare.sh gaussian 0.01
# ./prepare.sh sift100M 0.0001
# ./prepare.sh deep100M 0.0001

# ./prepare.sh sift1B 0.00001
# ./prepare.sh deep1B 0.00001

# ./memory.sh [dataset] [method] [R] [L] [sr]

# ./memory.sh enron l2 10 300
# ./memory.sh enron l2 16 300
# ./memory.sh enron l2 32 300
# ./memory.sh enron ls 10 300 0.01
# ./memory.sh enron ls 16 300 0.01
# ./memory.sh enron ls 32 300 0.01

# ./memory.sh gist l2 10 300
# ./memory.sh gist l2 16 300
# ./memory.sh gist l2 32 300
# ./memory.sh gist ls 10 300 0.01
# ./memory.sh gist ls 16 300 0.01
# ./memory.sh gist ls 32 300 0.01

# ./memory.sh glove1M l2 10 300
# ./memory.sh glove1M l2 16 300
# ./memory.sh glove1M l2 32 300
# ./memory.sh glove1M ls 10 300 0.01
# ./memory.sh glove1M ls 16 300 0.01
# ./memory.sh glove1M ls 32 300 0.01

# ./memory.sh glove2M l2 10 300
# ./memory.sh glove2M l2 16 300
# ./memory.sh glove2M l2 32 300
# ./memory.sh glove2M ls 10 300 0.01
# ./memory.sh glove2M ls 16 300 0.01
# ./memory.sh glove2M ls 32 300 0.01

# ./memory.sh imageNet l2 10 300
# ./memory.sh imageNet l2 16 300
# ./memory.sh imageNet l2 32 300
# ./memory.sh imageNet ls 10 300 0.01
# ./memory.sh imageNet ls 16 300 0.01
# ./memory.sh imageNet ls 32 300 0.01

# ./disk.sh [dataset] [method] [R] [L] [QD] [sr]

# ./disk.sh enron l2 10 300 32 0.25
# ./disk.sh enron l2 16 300 32 0.25
# ./disk.sh enron l2 32 300 32 0.25
# ./disk.sh enron ls 10 300 32 0.25 0.01
# ./disk.sh enron ls 16 300 32 0.25 0.01
# ./disk.sh enron ls 32 300 32 0.25 0.01
# ./disk.sh enron l2 10 300 32 0.5
# ./disk.sh enron l2 16 300 32 0.5
# ./disk.sh enron l2 32 300 32 0.5
# ./disk.sh enron ls 10 300 32 0.5 0.01
# ./disk.sh enron ls 16 300 32 0.5 0.01
# ./disk.sh enron ls 32 300 32 0.5 0.01

# ./disk.sh trevi l2 10 300 32 0.5
# ./disk.sh trevi l2 16 300 32 0.5
# ./disk.sh trevi l2 32 300 32 0.5
# ./disk.sh trevi ls 10 300 32 0.5 0.01
# ./disk.sh trevi ls 16 300 32 0.5 0.01
# ./disk.sh trevi ls 32 300 32 0.5 0.01

# ./disk.sh notre l2 10 300 16 0.5
# ./disk.sh notre l2 16 300 16 0.5
# ./disk.sh notre l2 32 300 16 0.5
# ./disk.sh notre ls 10 300 16 0.5 0.01
# ./disk.sh notre ls 16 300 16 0.5 0.01
# ./disk.sh notre ls 32 300 16 0.5 0.01

# ./disk.sh millionSong l2 10 300 32 0.5
# ./disk.sh millionSong l2 16 300 32 0.5
# ./disk.sh millionSong l2 32 300 32 0.5
# ./disk.sh millionSong ls 10 300 32 0.5 0.01
# ./disk.sh millionSong ls 16 300 32 0.5 0.01
# ./disk.sh millionSong ls 32 300 32 0.5 0.01

# ./disk.sh random l2 10 300 16 0.5
# ./disk.sh random l2 16 300 16 0.5
# ./disk.sh random l2 32 300 16 0.5
# ./disk.sh random ls 10 300 16 0.5 0.01
# ./disk.sh random ls 16 300 16 0.5 0.01
# ./disk.sh random ls 32 300 16 0.5 0.01

# ./disk.sh gaussian l2 10 300 16 0.5
# ./disk.sh gaussian l2 16 300 16 0.5
# ./disk.sh gaussian l2 32 300 16 0.5
# ./disk.sh gaussian ls 10 300 16 0.5 0.01
# ./disk.sh gaussian ls 16 300 16 0.5 0.01
# ./disk.sh gaussian ls 32 300 16 0.5 0.01

# ./disk.sh sift1M l2 10 300 16 0.5
# ./disk.sh sift1M l2 16 300 16 0.5
# ./disk.sh sift1M l2 32 300 16 0.5
# ./disk.sh sift1M ls 10 300 16 0.5 0.01
# ./disk.sh sift1M ls 16 300 16 0.5 0.01
# ./disk.sh sift1M ls 32 300 16 0.5 0.01

# ./disk.sh deep1M l2 10 300 16 0.5
# ./disk.sh deep1M l2 16 300 16 0.5
# ./disk.sh deep1M l2 32 300 16 0.5
# ./disk.sh deep1M ls 10 300 16 0.5 0.01
# ./disk.sh deep1M ls 16 300 16 0.5 0.01
# ./disk.sh deep1M ls 32 300 16 0.5 0.01

# ./disk.sh word2vec l2 10 300 32 0.5
# ./disk.sh word2vec l2 16 300 32 0.5
# ./disk.sh word2vec l2 32 300 32 0.5
# ./disk.sh word2vec ls 10 300 32 0.5 0.01
# ./disk.sh word2vec ls 16 300 32 0.5 0.01
# ./disk.sh word2vec ls 32 300 32 0.5 0.01

# ./disk.sh gist l2 10 300 32 0.5
# ./disk.sh gist l2 16 300 32 0.5
# ./disk.sh gist l2 32 300 32 0.5
# ./disk.sh gist ls 10 300 32 0.5 0.01
# ./disk.sh gist ls 16 300 32 0.5 0.01
# ./disk.sh gist ls 32 300 32 0.5 0.01
# ./disk.sh gist l2 10 300 32 1.5
# ./disk.sh gist l2 16 300 32 1.5
# ./disk.sh gist l2 32 300 32 1.5
# ./disk.sh gist ls 10 300 32 1.5 0.01
# ./disk.sh gist ls 16 300 32 1.5 0.01
# ./disk.sh gist ls 32 300 32 1.5 0.01

# ./disk.sh glove1M l2 10 300 16 0.5
# ./disk.sh glove1M l2 16 300 16 0.5
# ./disk.sh glove1M l2 32 300 16 0.5
# ./disk.sh glove1M ls 10 300 16 0.5 0.01
# ./disk.sh glove1M ls 16 300 16 0.5 0.01
# ./disk.sh glove1M ls 32 300 16 0.5 0.01
# ./disk.sh glove1M l2 10 300 16 0.6
# ./disk.sh glove1M l2 16 300 16 0.6
# ./disk.sh glove1M l2 32 300 16 0.6
# ./disk.sh glove1M ls 10 300 16 0.6 0.01
# ./disk.sh glove1M ls 16 300 16 0.6 0.01
# ./disk.sh glove1M ls 32 300 16 0.6 0.01

# ./disk.sh glove2M l2 10 300 32 1.0
# ./disk.sh glove2M l2 16 300 32 1.0
# ./disk.sh glove2M l2 32 300 32 1.0
# ./disk.sh glove2M ls 10 300 32 1.0 0.01
# ./disk.sh glove2M ls 16 300 32 1.0 0.01
# ./disk.sh glove2M ls 32 300 32 1.0 0.01
# ./disk.sh glove2M l2 10 300 32 2.0
# ./disk.sh glove2M l2 16 300 32 2.0
# ./disk.sh glove2M l2 32 300 32 2.0
# ./disk.sh glove2M ls 10 300 32 2.0 0.01
# ./disk.sh glove2M ls 16 300 32 2.0 0.01
# ./disk.sh glove2M ls 32 300 32 2.0 0.01
# ./disk.sh glove2M l2 10 300 32 3.0
# ./disk.sh glove2M l2 16 300 32 3.0
# ./disk.sh glove2M l2 32 300 32 3.0
# ./disk.sh glove2M ls 10 300 32 3.0 0.01
# ./disk.sh glove2M ls 16 300 32 3.0 0.01
# ./disk.sh glove2M ls 32 300 32 3.0 0.01

# ./disk.sh imageNet l2 10 300 16 1.0
# ./disk.sh imageNet l2 16 300 16 1.0
# ./disk.sh imageNet l2 32 300 16 1.0
# ./disk.sh imageNet ls 10 300 16 1.0 0.01
# ./disk.sh imageNet ls 16 300 16 1.0 0.01
# ./disk.sh imageNet ls 32 300 16 1.0 0.01
# ./disk.sh imageNet l2 10 300 16 1.5
# ./disk.sh imageNet l2 16 300 16 1.5
# ./disk.sh imageNet l2 32 300 16 1.5
# ./disk.sh imageNet ls 10 300 16 1.5 0.01
# ./disk.sh imageNet ls 16 300 16 1.5 0.01
# ./disk.sh imageNet ls 32 300 16 1.5 0.01

# ./disk.sh tiny5m l2 10 300 32 2.0
# ./disk.sh tiny5m l2 16 300 32 2.0
# ./disk.sh tiny5m l2 32 300 32 2.0
# ./disk.sh tiny5m ls 10 300 32 2.0 0.01
# ./disk.sh tiny5m ls 16 300 32 2.0 0.01
# ./disk.sh tiny5m ls 32 300 32 2.0 0.01

# ./disk.sh deep10M l2 10 300 16 4.0
# ./disk.sh deep10M l2 16 300 16 4.0
# ./disk.sh deep10M l2 32 300 16 4.0
# ./disk.sh deep10M ls 10 300 16 4.0 0.001
# ./disk.sh deep10M ls 16 300 16 4.0 0.001
# ./disk.sh deep10M ls 32 300 16 4.0 0.001

# ./disk.sh sift10M l2 10 300 16 4.0
# ./disk.sh sift10M l2 16 300 16 4.0
# ./disk.sh sift10M l2 32 300 16 4.0
# ./disk.sh sift10M ls 10 300 16 4.0 0.001
# ./disk.sh sift10M ls 16 300 16 4.0 0.001
# ./disk.sh sift10M ls 32 300 16 4.0 0.001

# ./disk.sh spacev10M l2 10 300 16 4.0
# ./disk.sh spacev10M l2 16 300 16 4.0
# ./disk.sh spacev10M l2 32 300 16 4.0
# ./disk.sh spacev10M ls 10 300 16 4.0 0.001
# ./disk.sh spacev10M ls 16 300 16 4.0 0.001
# ./disk.sh spacev10M ls 32 300 16 4.0 0.001

# ./disk.sh tiny80M l2 10 300 16 8.0
# ./disk.sh tiny80M l2 16 300 16 8.0
# ./disk.sh tiny80M l2 32 300 16 8.0
# ./disk.sh tiny80M ls 10 300 16 8.0 0.001
# ./disk.sh tiny80M ls 16 300 16 8.0 0.001
# ./disk.sh tiny80M ls 32 300 16 8.0 0.001

# ./disk.sh random_hypercube l2 10 300 16 0.5
# ./disk.sh random_hypercube l2 16 300 16 0.5
# ./disk.sh random_hypercube l2 32 300 16 0.5
# ./disk.sh random_hypercube ls 10 300 16 0.5 0.01
# ./disk.sh random_hypercube ls 16 300 16 0.5 0.01
# ./disk.sh random_hypercube ls 32 300 16 0.5 0.01

# ./disk.sh sift100M l2 10 300 16 16
# ./disk.sh sift100M ls 10 300 16 16 0.0001

# ./disk.sh deep100M l2 10 300 16 16
# ./disk.sh deep100M ls 10 300 16 16 0.0001

# ./disk.sh sift1B l2 10 300 16 32
# ./disk.sh sift1B ls 10 300 16 32 0.00001

# ./disk.sh deep1B l2 10 300 16 32
# ./disk.sh deep1B ls 10 300 16 32 0.00001