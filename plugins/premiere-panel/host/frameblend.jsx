// ExtendScript side of Frame Blend for Premiere, run by the hidden companion
// (auto.html). Frame blending is on by default: every clip that plays more
// frames per second than its sequence shows (frame rate x speed) gets the
// Frame Blend effect with its Time Interpolation set to Frame Blending. The
// clip's Time Interpolation is the switch (right-click > Time Interpolation,
// also for many selected clips at once): Frame Blending blends, Frame Sampling
// and Optical Flow leave the clip to Premiere. Clips that already have the
// effect are never changed again, so that choice stays; the first time a
// sequence is seen, all its clips are set up once and the sequence is marked
// in the project file.

var FB_NAME = "Frame Blend";
var FB_MATCH = "AE.FrameBlend Native";
var FB_TICKS_PER_SECOND = 254016000000;
var FB_FRAME_BLENDING = 1;  // QE time interpolation type (0 Frame Sampling, 2 Optical Flow)
var FB_NS = "http://ns.adobe.com/premierePrivateProjectMetaData/1.0/";
var FB_FIELD = "FrameBlendSetUp";

// Media frame rate per project item (footage interpretation), by node id.
if (typeof $.global.fbRates === "undefined") {
    $.global.fbRates = {};
}

// ---- sequences set up once, marked in the project file ----

function fbXmp() {
    if (ExternalObject.AdobeXMPScript === undefined) {
        ExternalObject.AdobeXMPScript = new ExternalObject("lib:AdobeXMPScript");
    }
}

function fbIsSetUp(seq) {
    try {
        fbXmp();
        var value = new XMPMeta(seq.projectItem.getProjectMetadata()).getProperty(FB_NS, FB_FIELD);
        return value !== undefined && value !== null && String(value) === "1";
    } catch (e) {
        return false;
    }
}

function fbMarkSetUp(seq) {
    fbXmp();
    app.project.addPropertyToProjectMetadataSchema(FB_FIELD, "Frame Blend set up", 2);
    var meta = new XMPMeta(seq.projectItem.getProjectMetadata());
    meta.setProperty(FB_NS, FB_FIELD, "1");
    seq.projectItem.setProjectMetadata(meta.serialize(), [FB_FIELD]);
}

// ---- clips ----

function fbHasEffect(clip) {
    var components = clip.components;
    for (var i = 0; i < components.numItems; i++) {
        if (components[i].matchName === FB_MATCH || components[i].displayName === FB_NAME) {
            return true;
        }
    }
    return false;
}

function fbMediaRate(item) {
    var id = item.nodeId;
    if ($.global.fbRates[id] === undefined) {
        var rate = 0;
        try {
            rate = item.getFootageInterpretation().frameRate;
        } catch (e) {}
        $.global.fbRates[id] = rate;
    }
    return $.global.fbRates[id];
}

// Video clips from media files that play more frames per second than the
// sequence shows; nested sequences, adjustment layers and stills are skipped.
function fbEligible(clip, sequenceRate) {
    var item = clip.projectItem;
    if (!item) {
        return false;
    }
    try {
        if (item.isSequence() || !item.getMediaPath()) {
            return false;
        }
    } catch (e) {
        return false;
    }
    var speed = 1;
    try {
        speed = Math.abs(clip.getSpeed());
    } catch (e2) {}
    return fbMediaRate(item) * speed > sequenceRate * 1.001;
}

// QE items of the active sequence's track, by start time (QE lists gaps too).
function fbQeItems(trackIndex) {
    var byStart = {};
    var qeTrack = qe.project.getActiveSequence().getVideoTrackAt(trackIndex);
    for (var q = 0; q < qeTrack.numItems; q++) {
        var qeItem = qeTrack.getItemAt(q);
        if (qeItem && qeItem.type !== "Empty") {
            byStart[String(qeItem.start.ticks)] = qeItem;
        }
    }
    return byStart;
}

// Gives eligible clips of the active sequence the effect and Frame Blending:
// all of them when `everything`, otherwise only those without the effect.
function fbSetUp(seq, everything) {
    app.enableQE();
    var effect = null;
    var rate = FB_TICKS_PER_SECOND / parseFloat(seq.timebase);
    for (var t = 0; t < seq.videoTracks.numTracks; t++) {
        var track = seq.videoTracks[t];
        var qeByStart = null;
        for (var i = 0; i < track.clips.numItems; i++) {
            var clip = track.clips[i];
            var has = fbHasEffect(clip);
            if ((has && !everything) || !fbEligible(clip, rate)) {
                continue;
            }
            qeByStart = qeByStart || fbQeItems(t);
            var qeItem = qeByStart[String(clip.start.ticks)];
            if (!qeItem) {
                continue;
            }
            if (!has) {
                effect = effect || qe.project.getVideoEffectByName(FB_NAME);
                if (!effect) {
                    return;
                }
                qeItem.addVideoEffect(effect);
            }
            // QE is undocumented and its time interpolation calls may be
            // missing from older Premiere; the effect then blends regardless
            // (the GPU filter blends when the setting is not reported).
            try {
                if (qeItem.timeInterpolationType !== FB_FRAME_BLENDING) {
                    qeItem.setTimeInterpolationType(FB_FRAME_BLENDING);
                }
            } catch (e) {}
        }
    }
}

// Cheap signature of the active sequence: id, frame rate and clip count per
// video track. The frame rate is part of it because a sequence made with New
// Sequence From Clip starts at the clip's rate (nothing to blend) and is often
// switched to 30 or 60 fps afterwards; its clips must be set up then.
function fbSignature() {
    var seq = app.project ? app.project.activeSequence : null;
    if (!seq) {
        return "";
    }
    var s = seq.sequenceID + "@" + seq.timebase;
    for (var t = 0; t < seq.videoTracks.numTracks; t++) {
        s += ":" + seq.videoTracks[t].clips.numItems;
    }
    return s;
}

// Polled by the hidden companion with the signature it saw last; when the
// timeline changed, new clips are set up (a sequence seen for the first time
// is set up entirely). Returns the current signature.
function fbAutoApply(lastSignature) {
    var signature = fbSignature();
    if (signature === "" || signature === lastSignature) {
        return signature;
    }
    var seq = app.project.activeSequence;
    // A failure (a scripting call this Premiere version lacks) is not retried
    // every poll: the signature is returned either way.
    try {
        if (fbIsSetUp(seq)) {
            fbSetUp(seq, false);
        } else {
            fbSetUp(seq, true);
            fbMarkSetUp(seq);
        }
    } catch (e) {}
    return fbSignature();
}
