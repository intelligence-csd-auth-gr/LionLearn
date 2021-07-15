% The possible importances can be Positive, Negative or Neutral
importance('Positive').
importance('Negative').
importance('Neutral').

alteration('Increase').
alteration('Decrease').
alteration('Stable').

sign('+').
sign('-').

feature(X):-feature_importance(X, _).

untruthful(X) :- feature(X).

untrusted(X) :- trusted(X),!, fail.
untrusted('Explanation').

modify(X,Y) :- feature(X),importance(Y),feature_importance(X, Y).

modification(X,Y,Z) :- modify(X,Y),importance(Y),sign(Z).

expected(X,'Positive','+','Increase') :-
    evaluated(X,'Positive','+','Increase'),modification(X,'Positive','+').

expected(X,'Positive','-','Decrease') :-
    evaluated(X,'Positive','-','Decrease'),modification(X,'Positive','-').

expected(X,'Negative','+','Decrease') :-
    evaluated(X,'Negative','+','Decrease'),modification(X,'Negative','+').

expected(X,'Negative','-','Increase') :-
    evaluated(X,'Negative','-','Increase'),modification(X,'Negative','-').

expected(X,'Neutral','+','Stable') :-
    evaluated(X,'Neutral','+','Stable'),modification(X,'Neutral','+').

expected(X,'Neutral','-','Stable') :-
    evaluated(X,'Neutral','-','Stable'),modification(X,'Neutral','-').

truthful(X) :-  expected(X,'Positive','+','Increase'),expected(X,'Positive','-','Decrease').
truthful(X) :-  expected(X,'Negative','+','Decrease'),expected(X,'Negative','-','Increase').
truthful(X) :- expected(X,'Neutral','+','Stable'),expected(X,'Neutral','-','Stable').

%truthful(X) :- untruthful(X), !, fail.
not(untruthful(X)) :- truthful(X),!, display(X," is not untruthful").
not(untruthful(X)) :- display(X, " is indeed untruthful").

display( Message, Goal)  :-
write( Message),
write( Goal), nl.
feature_importance('F42', 'Negative').
feature_importance('F43', 'Positive').
feature_importance('F44', 'Negative').
feature_importance('F45', 'Negative').
feature_importance('F46', 'Positive').
feature_importance('F47', 'Positive').
feature_importance('F48', 'Negative').
feature_importance('F49', 'Negative').
feature_importance('F50', 'Positive').
evaluated('F42','Negative','+','Decrease').
evaluated('F42','Negative','-','Increase').
evaluated('F43','Positive','+','Increase').
evaluated('F43','Positive','-','Decrease').
evaluated('F44','Negative','+','Decrease').
evaluated('F44','Negative','-','Increase').
evaluated('F45','Negative','+','Decrease').
evaluated('F45','Negative','-','Increase').
evaluated('F46','Positive','+','Increase').
evaluated('F46','Positive','-','Decrease').
evaluated('F47','Positive','+','Increase').
evaluated('F48','Negative','+','Decrease').
evaluated('F48','Negative','-','Increase').
evaluated('F49','Negative','+','Decrease').
evaluated('F49','Negative','-','Increase').
evaluated('F50','Positive','+','Increase').
trusted('Explanation') :- not(untruthful('F42')), writeln('F42 is untruthful'), not(untruthful('F43')), writeln('F43 is untruthful'), not(untruthful('F44')), writeln('F44 is untruthful'), not(untruthful('F45')), writeln('F45 is untruthful'), not(untruthful('F46')), writeln('F46 is untruthful'), not(untruthful('F47')), writeln('F47 is untruthful'), not(untruthful('F48')), writeln('F48 is untruthful'), not(untruthful('F49')), writeln('F49 is untruthful'), not(untruthful('F50')), writeln('F50 is untruthful').
