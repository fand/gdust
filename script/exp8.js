#!/usr/bin/env node
// Usage: $ node exp8.js DATA

var fs = require('fs');
var sysPath = require('path');
var exec = require('child_process').exec;
var file_raw = process.argv[2];
var file = process.argv[3];
var topk = +process.argv[4];

if (process.argv.length !== 5) {
  console.log('wrong arguments');
  process.exit();
}

var Cooler = require('./Cooler');

var matches = {};
var counts = {};

var exp = function (num) {
  if (num === 200) {
    finish();
    return;
  }

  var cmd = "bin/gdustdtw --exp 8 " + file_raw + ' ' + file + ' --target ' + num + ' --topk ' + topk;

  exec(cmd, function (err, stdout, stderr) {
    if (err) { throw err; }

    // Parse results
    var result = JSON.parse(stdout);

    // save matches
    for (var k in result) {
      if (matches[k] == null) { matches[k] = []; }
      matches[k].push(result[k]);
    }

    // count matches
    var TRUTH = result.TRUTH;
    for (var k in result) {
      if (counts[k] == null) { counts[k] = []; }
      var count = 0;
      result[k].forEach(function (r, i) {
        if (TRUTH.indexOf(r) !== -1) { count++; }
      });
      counts[k].push(count);
    }

    console.log('############################ num: ', num);
    console.log(stdout);
    console.error(stderr);

    Cooler(function () {
      exp(num + 1);
    });
  });
};

var finish = function () {
  console.log('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> matches');
  console.log(JSON.stringify(matches));
  console.log('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> counts');
  console.log(JSON.stringify(counts));
  console.log('END');
};


// DO IT!
exp(0);
