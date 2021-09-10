## Setting up Virtuoso RDF storage

Documentation is [here](http://vos.openlinksw.com/owiki/wiki/VOS).

### Get the sources

```bash
git clone git@github.com:openlink/virtuoso-opensource.git
cd virtuoso-opensource
git checkout 01a82b2d4103a816c280d6e60727b64beabc7674 # we used this commit but should work on others as well
```

### Compile and install

Check [the documentation](https://github.com/openlink/virtuoso-opensource) for dependencies and instructions for different systems.

For us, it worked easy:
```bash
export VIRTUOSO_PATH=$HOME/virtuoso/ # path to install Virtuoso - can be anywhere
./autogen.sh
./configure --prefix=$VIRTUOSO_PATH
make
make install
```

### Finish setting it up

Find a free port on your machine, e.g., with this (e.g. 38691):
```bash
export VIRTUOSO_PORT=`$QDMR_ROOT/utils/get_free_port.sh`
echo $VIRTUOSO_PORT 
```

Edit config of Virtuoso: $VIRTUOSO_PATH/var/lib/virtuoso/db/virtuoso.ini
1. Add to DirsAllowed (line 66) the full path for path cached rdf graphs (e.g., obtained from `echo $QDMR_ROOT/data/spider/database_rdf`)
2. Change ServerPort (line 121) to the free port, e.g. 38691)
3. (optional) You can increase resource usage of the server for speed (e.g., MaxQueryMem and MaxClientConnections) and uncomment appropriate values for NumberOfBuffers and MaxDirtyBuffers.

Prepare data as serialized RDF graphs (might take some time):
```bash
mkdir -p $QDMR_ROOT/data/spider/database_rdf
python $QDMR_ROOT/qdmr2sparql/serialize_rdf_graphs.py
```

### Launch the Virtuoso server

```bash
$VIRTUOSO_PATH/bin/virtuoso-t -fd +configfile $VIRTUOSO_PATH/var/lib/virtuoso/db/virtuoso.ini
```
It should print something like this:
```
14:17:06 INFO: HTTP/WebDAV server online at 38691
14:17:06 INFO: Server online at 1111 (pid 4784)
```
indicating the port for HTTP access (38691) and the port for the `isql` command (1111).

Remember the machine where the server is launched, e.g., by this: 
```bash
export $VIRTUOSO_HOST=hostname
echo $VIRTUOSO_HOST
```

The SPARQL endpoint for Virtuoso would come from this (e.g., "http://localhost:38691/sparql/" - put this link to the configs in `text2qdmr/configs/experiments`):
```bash
echo "http://${VIRTUOSO_HOST}:${VIRTUOSO_PORT}/sparql/"
```

Check  that isql works (show all graphs):
```bash
$VIRTUOSO_PATH/bin/isql exec="DB.DBA.SPARQL_SELECT_KNOWN_GRAPHS();"
```

You can also run isql over network when the server is launched on the main node:
```bash
$VIRTUOSO_PATH/bin/isql $VIRTUOSO_HOST:1111 exec="DB.DBA.SPARQL_SELECT_KNOWN_GRAPHS();"
```

### Load the data into Virtuoso

These steps should be done once when setting up the server.
It should load all the data automatically when relaunched.

Prepare batch loading:
```bash
$VIRTUOSO_PATH/bin/isql exec="ld_dir ('${QDMR_ROOT}/data/spider/database_rdf', '*.ttl', NULL);"
```

Check that the files were found. This should print a long table aff all the databases from `$QDMR_ROOT/data/spider/database_rdf`.
```bash
$VIRTUOSO_PATH/bin/isql exec="select * from DB.DBA.load_list;"
```

Run the loading (might take some time):
```bash
$VIRTUOSO_PATH/bin/isql exec="rdf_loader_run();checkpoint;"
```

Check that the graphs were loaded. This should print the names of all the loaded graphs.
```bash
$VIRTUOSO_PATH/bin/isql exec="DB.DBA.SPARQL_SELECT_KNOWN_GRAPHS();"
```

If you want to delete everything from Virtuoso and start over run these two commands:
```bash
$VIRTUOSO_PATH/bin/isql exec="delete from DB.DBA.load_list;"
$VIRTUOSO_PATH/bin/isql exec="delete from DB.DBA.RDF_QUAD ;"
```
