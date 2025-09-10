// Package jsh provides a slog.Handler that emits JSON using the Go 1.25
// encoding/json/v2 + encoding/json/jsontext stack, with optional pretty printing.
//
// This adheres to slog.NewJSONHandler’s built-ins exactly:
//   - Keys: "time", "level", "msg", "source" (in that order, then user attrs).
//   - Values:
//     time   -> time.Time (omitted if zero)
//     level  -> slog.Level (marshaled as its text, e.g., "INFO")
//     msg    -> string
//     source -> *slog.Source (object with fields: function,file,line)
//   - ReplaceAttr is invoked for every non-group attribute, including built-ins,
//     with the current group path, per HandlerOptions docs.
//
// Requires Go 1.25 with GOEXPERIMENT=jsonv2.
package jsonv2handler

import (
	"bytes"
	"context"
	"io"
	"log/slog"
	"sync"

	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
)

// Options configures the handler.
type Options struct {
	// Minimum level to log; if nil, all are enabled.
	Level slog.Leveler

	// Indent controls pretty printing when non-empty (e.g., "  " or "\t").
	Indent string

	// Built-in key customizations. Defaults match slog constants.
	TimeKey, LevelKey, MessageKey, SourceKey string

	// Time format is ONLY used if you change how you render time yourself.
	// We pass the time.Time value through to JSON (same as slog.JSONHandler).
	TimeFormat string // unused by default

	// Include source info.
	AddSource bool

	// ReplaceAttr is called for every non-group attribute (built-ins included).
	ReplaceAttr func(groups []string, a slog.Attr) slog.Attr
}

// JSONHandler implements slog.Handler and writes JSON.
type JSONHandler struct {
	w     io.Writer
	opts  Options
	mu    *sync.Mutex // shared across clones
	group []string    // accumulated WithGroup()
	attrs []slog.Attr // accumulated WithAttrs()
}

func New(w io.Writer, opts *Options) *JSONHandler {
	var o Options
	if opts != nil {
		o = *opts
	}
	// Default keys to slog’s built-in constants.
	if o.TimeKey == "" {
		o.TimeKey = slog.TimeKey
	}
	if o.LevelKey == "" {
		o.LevelKey = slog.LevelKey
	}
	if o.MessageKey == "" {
		o.MessageKey = slog.MessageKey
	}
	if o.SourceKey == "" {
		o.SourceKey = slog.SourceKey
	}
	return &JSONHandler{
		w:    w,
		opts: o,
		mu:   &sync.Mutex{},
	}
}

func (h *JSONHandler) Enabled(_ context.Context, level slog.Level) bool {
	if h.opts.Level == nil {
		return true
	}
	return level >= h.opts.Level.Level()
}

func (h *JSONHandler) Handle(_ context.Context, r slog.Record) error {
	// Snapshot state outside the write lock.
	baseGroups := append([]string(nil), h.group...)
	baseAttrs := append([]slog.Attr(nil), h.attrs...)

	// Build an ordered object: time, level, msg, source, then attrs.
	root := newObj()

	// Helper for built-ins at root path with ReplaceAttr.
	putBuiltin := func(a slog.Attr) {
		// Values for built-ins are already concrete (no LogValuer).
		if h.opts.ReplaceAttr != nil {
			a = h.opts.ReplaceAttr(baseGroups, a)
		}
		if a.Key != "" {
			root.putKV(a.Key, valueToNode(a.Value))
		}
	}

	// time (omit if zero)
	if h.opts.TimeKey != "" && !r.Time.IsZero() {
		putBuiltin(slog.Attr{Key: h.opts.TimeKey, Value: slog.TimeValue(r.Time)})
	}

	// level (as slog.Level value, not string; JSON shows its text)
	if h.opts.LevelKey != "" {
		putBuiltin(slog.Attr{Key: h.opts.LevelKey, Value: slog.AnyValue(r.Level)})
	}

	// msg
	if h.opts.MessageKey != "" && r.Message != "" {
		putBuiltin(slog.Attr{Key: h.opts.MessageKey, Value: slog.StringValue(r.Message)})
	}

	// source (*slog.Source) when requested and available
	if h.opts.AddSource && h.opts.SourceKey != "" {
		if src := r.Source(); src != nil {
			putBuiltin(slog.Attr{Key: h.opts.SourceKey, Value: slog.AnyValue(src)})
		}
	}

	// Accumulated WithAttrs then record attrs, preserving insertion order.
	builder := &attrBuilder{
		opts:       h.opts,
		root:       root,
		rootGroups: baseGroups,
	}
	for _, a := range baseAttrs {
		builder.addAttr(a)
	}
	r.Attrs(func(a slog.Attr) bool {
		builder.addAttr(a)
		return true
	})

	// Encode with optional pretty printing via jsontext.
	var buf bytes.Buffer
	var encOpts []jsontext.Options
	if h.opts.Indent != "" {
		encOpts = append(encOpts, jsontext.Multiline(true), jsontext.WithIndent(h.opts.Indent))
	}
	enc := jsontext.NewEncoder(&buf, encOpts...)
	if err := root.encode(enc); err != nil {
		return err
	}

	// Single atomic write.
	h.mu.Lock()
	_, err := h.w.Write(buf.Bytes())
	if err == nil {
		_, err = io.WriteString(h.w, "\n")
	}
	h.mu.Unlock()
	return err
}

func (h *JSONHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	nh := *h
	nh.attrs = append(append([]slog.Attr{}, h.attrs...), sanitizeAttrs(attrs)...)
	nh.mu = h.mu // share the same lock
	return &nh
}

func (h *JSONHandler) WithGroup(name string) slog.Handler {
	if name == "" {
		return h
	}
	nh := *h
	nh.group = append(append([]string{}, h.group...), name)
	nh.mu = h.mu // share the same lock
	return &nh
}

// ---------- ordered object & encoding ----------

type node struct {
	obj  *orderedObj // if non-nil, an object
	leaf any         // otherwise, a leaf value (marshaled by json/v2)
}

type pair struct {
	key string
	val node
}

type orderedObj struct {
	pairs []pair
}

func newObj() *orderedObj { return &orderedObj{pairs: make([]pair, 0)} }

func (o *orderedObj) getOrCreate(key string) *orderedObj {
	for i := range o.pairs {
		if o.pairs[i].key == key {
			if o.pairs[i].val.obj == nil {
				o.pairs[i].val.obj = newObj()
			}
			return o.pairs[i].val.obj
		}
	}
	ch := newObj()
	o.pairs = append(o.pairs, pair{key: key, val: node{obj: ch}})
	return ch
}

func (o *orderedObj) putKV(key string, val node) {
	for i := range o.pairs {
		if o.pairs[i].key == key {
			o.pairs[i].val = val
			return
		}
	}
	o.pairs = append(o.pairs, pair{key: key, val: val})
}

func (o *orderedObj) encode(enc *jsontext.Encoder) error {
	if err := enc.WriteToken(jsontext.BeginObject); err != nil {
		return err
	}
	for i := range o.pairs {
		if err := enc.WriteToken(jsontext.String(o.pairs[i].key)); err != nil {
			return err
		}
		if o.pairs[i].val.obj != nil {
			if err := o.pairs[i].val.obj.encode(enc); err != nil {
				return err
			}
			continue
		}
		if err := jsonv2.MarshalEncode(enc, o.pairs[i].val.leaf); err != nil {
			return err
		}
	}
	return enc.WriteToken(jsontext.EndObject)
}

// ---------- slog value handling ----------

func sanitizeAttrs(in []slog.Attr) []slog.Attr {
	out := make([]slog.Attr, 0, len(in))
	for _, a := range in {
		if a.Key == "" {
			continue
		}
		// Resolve only the value (there is no Attr.Resolve in slog).
		a.Value = a.Value.Resolve()
		out = append(out, a)
	}
	return out
}

type attrBuilder struct {
	opts       Options
	root       *orderedObj
	rootGroups []string
}

func (b *attrBuilder) addAttr(a slog.Attr) {
	if a.Key == "" {
		return
	}
	a.Value = a.Value.Resolve()
	switch a.Value.Kind() {
	case slog.KindGroup:
		ng := append(append([]string(nil), b.rootGroups...), a.Key)
		for _, m := range a.Value.Group() {
			b.addWithPath(ng, m)
		}
	default:
		b.addWithPath(b.rootGroups, a)
	}
}

func (b *attrBuilder) addWithPath(path []string, a slog.Attr) {
	if b.opts.ReplaceAttr != nil {
		a = b.opts.ReplaceAttr(path, a)
	}
	if a.Key == "" {
		return // dropped
	}
	cur := b.root
	for _, g := range path {
		cur = cur.getOrCreate(g)
	}
	a.Value = a.Value.Resolve()
	if a.Value.Kind() == slog.KindGroup {
		cur.putKV(a.Key, groupToNode(a.Value.Group()))
		return
	}
	cur.putKV(a.Key, valueToNode(a.Value))
}

func groupToNode(members []slog.Attr) node {
	obj := newObj()
	for _, a := range members {
		if a.Key == "" {
			continue
		}
		a.Value = a.Value.Resolve()
		if a.Value.Kind() == slog.KindGroup {
			obj.putKV(a.Key, groupToNode(a.Value.Group()))
		} else {
			obj.putKV(a.Key, valueToNode(a.Value))
		}
	}
	return node{obj: obj}
}

func valueToNode(v slog.Value) node {
	v = v.Resolve()
	switch v.Kind() {
	case slog.KindString:
		return node{leaf: v.String()}
	case slog.KindInt64:
		return node{leaf: v.Int64()}
	case slog.KindUint64:
		return node{leaf: v.Uint64()}
	case slog.KindFloat64:
		return node{leaf: v.Float64()}
	case slog.KindBool:
		return node{leaf: v.Bool()}
	case slog.KindDuration:
		return node{leaf: v.Duration().String()}
	case slog.KindTime:
		return node{leaf: v.Time()}
	case slog.KindAny:
		return node{leaf: v.Any()}
	case slog.KindGroup:
		return groupToNode(v.Group())
	case slog.KindLogValuer:
		return valueToNode(v.Resolve())
	default:
		return node{leaf: v.String()}
	}
}
