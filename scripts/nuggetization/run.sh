export JAVA_HOME=${YOUR_JAVA_DIR}/jdk-21.0.7
export CLASSPATH=.:${JAVA_HOME}/lib
export PATH=${CLASSPATH}:${JAVA_HOME}/bin:$PATH
export JVM_PATH=${YOUR_JAVA_DIR}/jdk-21.0.7/lib/server/libjvm.so
# replace YOUR_JAVA_DIR with your actual Java directory

python src/nuggetize.py
