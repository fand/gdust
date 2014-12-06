#!/usr/bin/env node

var fs = require('fs');
var USAGE = 'Usage: hoge';

if (process.argv.length !== 3 && process.argv.length !== 4) {
  console.error('Wrong arguments!');
  console.error(USAGE);
  process.exit();
}

var filename = process.argv[2];
if (!fs.existsSync(filename)) {
  console.error('File "' + filename + '" doesn\'t exist!');
  console.error(USAGE);
}

var outfile = process.argv[3] || 'out.json';

var match = function (str, regexp, dst) {
  var m = str.match(regexp);
  return m ? [].slice.call(m, 1) : [dst];
};

fs.readFile(filename, 'utf8', function (err, data) {
  if (err) { throw err; }
  var lines = data.split(/\r|\n|\r\n|\n\r/);

  var k = 0;
  var i = 0;
  var opts = [];
  lines.forEach(function (line) {
    var m = match(line, /top k: (\d+)/, k)[0];
    i = match(line, /ts : (\d+)/, i)[0];
    opts[i] = opts[i] || {};

    var kv = match(line, /^(\S*)\s*:\s*(.*)$/, null);
    if (kv[0]) {
      var key = kv[0];
      opts[i][key] = opts[i][key] || [];

      var vals = kv[1].split(', ');
      vals.forEach(function (val) {
        if (val.match(/\S/)) { opts[i][key].push(+val); }
      });
    }
  });

  verify(opts, k);
});

var verify = function (opts, k) {
  var counts = {};
  opts.forEach(function (opt, i) {
    var basis = {};
    opt.eucl.forEach(function (v) {
      basis[v] = true;
    });
    for (var key in opt) {
      if (key === 'eucl') { continue; }
      counts[key] = counts[key] || {
        x: [], y:[], type: "bar", name: key
      };
      var count = 0;
      opt[key].forEach(function (v) {
        if (basis[v]) count++;
      });
      counts[key].x.push(i);
      counts[key].y.push(count);
    }
  });
  var results = [];
  for (var key in counts) {
    results.push(counts[key]);
  }

  fs.writeFile(outfile, JSON.stringify(results), 'utf8', function () {
    console.log('yo');
  });
};
