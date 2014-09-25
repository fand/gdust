#!/usr/bin/env node

var fs = require('fs');
var path = require('path');
var commander = require('commander');

// Parse options.
commander
    .version('0.0.1')
    .option('-s, --src [path]', 'Path for input')
    .option('-d, --dst [path]', 'Path for output')
    .option('-l, --limit <num>', 'Num of lines')
    .parse(process.argv);

const CWD = process.cwd();
const DATA_DIR =  path.join(CWD, (commander.src || 'data'));
const SMALL_DIR =  path.join(CWD, (commander.dst || 'data/small'));
const LIMIT = commander.limit || 10;

var forFilesIn = function (dirpath, callback) {
    fs.readdir(dirpath, function (err, files) {
        if (err) { throw err; }
        files.forEach(callback);
    });
};

var data2lines = function (data) {
    return data.toString().split(/\n|\r|\n\r/);
};

var limitLines = function (lines) {
    var limit_min = Math.min(LIMIT, lines.length);
    return lines.slice(0, limit_min);
};

// main
forFilesIn(DATA_DIR, function (file) {
    var stat = fs.statSync(path.join(DATA_DIR, file));
    if (! stat.isFile()) { return; }

    var fullpath = path.join(DATA_DIR, file);
    fs.readFile(fullpath, 'utf8', function (err, data) {
        if (err) { throw err; }
        var lines = data2lines(data);
        var lines_limited = limitLines(lines);
        var data_out = lines_limited.join('\n') + '\n';
        var dst = path.join(SMALL_DIR, file);
        fs.writeFile(dst, data_out, 'utf8', function(err){ if (err) throw err; });
    });
});
